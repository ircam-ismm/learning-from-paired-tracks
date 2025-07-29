from architecture.Model import load_model_checkpoint
from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
from utils.utils import lock_gpu, prGreen, prRed, prYellow, detect_onsets, find_non_empty, compute_consecutive_lengths
from .metrics import compute_accuracy as compute_acc
import torch
import numpy as np
import sys, typing, os
from typing import Union, List, Optional
import scipy.io.wavfile as wav
import soundfile as sf
from librosa import resample
from munch import Munch
from concatenate import Concatenator, TimeStamp #custom library for concatenating audio chunks from time markers
import time
from pathlib import Path
#for dicy2 library
sys.path.insert(0,"../Dicy2-python")

from MusicDataset.MusicDataset_v2 import MusicContainer4dicy2,Fetcher,MusicDataCollator
from torch.utils.data import DataLoader

from dicy2.corpus_event import Dicy2CorpusEvent # type: ignore
from dicy2.generator import Dicy2Generator # type: ignore
from dicy2.label import ListLabel # type: ignore
from dicy2.prospector import FactorOracleProspector # type: ignore
from gig.main.corpus import GenericCorpus # type: ignore
from gig.main.influence import LabelInfluence # type: ignore
from gig.main.query import InfluenceQuery # type: ignore

# import logging

# verbose = True
# log_level: int = logging.DEBUG if verbose else logging.INFO
# logging.basicConfig(level=log_level, format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(name)s: %(message)s',
#                         datefmt="%H:%M:%S")

def generate_memory_corpus(memory_ds : MusicContainer4dicy2, model : Seq2SeqCoupling, chunk_segmentation : str, batch_size : int):
    
    #dicy2 args
    max_continuity: int = 100000  # longest continuous sequence of the original text that is allowed before a jump is forced
    force_output: bool = False #True  # if no matches are found: output the next event (if True) or output None (if False)
    label_type = ListLabel
    
    collate_fn = MusicDataCollator(unifrom_chunks=chunk_segmentation!="onset")
    memory_loader = DataLoader(memory_ds,batch_size,False,collate_fn=collate_fn)
    memory_fetcher = Fetcher(memory_loader, model.device)
    
    #generate the whole corpus data from memory chunks
    labels = []
    corpus_data=[]
    all_memory_chunks = []
    last_slice=0
    native_chunk_idx = 0
    
    sliding = memory_ds.pre_segmentation_strategy =='sliding' #if we generate memory using sliding windows (using context of previous chunk)
    
    if sliding:
        output_hop_size = memory_ds.hop_size/memory_ds.max_duration #the equivalent hop size in chunks
        # print(output_hop_size,memory_ds.hop_size,memory_ds.max_duration)
        # if not output_hop_size.is_integer():
        #     raise RuntimeError("If hop size is not a multiple of chunk size there wont be an integer number of chunks equivalent to hop_size...")
        
        output_hop_size=int(output_hop_size)
        print(memory_ds.hop_size,output_hop_size)
    
    for i in range(len(memory_fetcher)):
        memory_data = next(memory_fetcher) #contains chunks and other data for model
        #encode memory -> extract labels for each slice/chunk
        #memory_idx = model.encoder.forward(memory_data.src, padding_mask = memory_data.src_padding_masks[0])[1] #(B,S)
        memory_idx, memory_idx_special_tokens = model.encode(memory_data.src,memory_data.src_padding_masks,both=True)[1] #memory_idx with and without special tokens <sos> <eos> <pad> (B,S)
        
        #corpus = memory as [(label, content)] where label is codebook index from encoding and content is the slice index
        
        #TODO : OPTIMIZE THIS CODE PART
        #iterate over batch elements
        for j,idxs in enumerate(memory_idx):
            first = (i==0 and j==0) #first segment, could be changed with a flag init at first = True and here "if first==True:first=False"
            
            slices = np.arange(len(idxs))
            
            #retrieve corresponding memory chunks
            memory_chunks=memory_ds.get_native_chunks(native_chunk_idx) #unprocessed chunks with native sr
            print("memory chunk and idxs len",len(memory_chunks), len(idxs))
            print("slices", slices)
            
            if sliding:
                if not first: #except for first chunk of first batch

                    idxs = idxs[-output_hop_size:] #only keep the continuation of context (given by previous chunk)
                    
                    slices = slices[-output_hop_size:] #only keep actual slices (rmeove context)
                    slices = slices - slices[0] #shift left slices
                    
                    memory_chunks = memory_chunks[-output_hop_size:] #crop memeory chunks too
            
            print(idxs,slices)
            corpus_data.extend([(label.item(), last_slice+slice) for label,slice in zip(idxs,slices)])
            last_slice = corpus_data[-1][1]+1 #slices only go from 0 to max so we need to update the real slice index basded on ietration
        
            all_memory_chunks.extend(memory_chunks) #append to list of all chunks
            
            native_chunk_idx+=1
        
        print(memory_idx.shape, memory_idx)
        labels.extend(memory_idx_special_tokens.reshape(-1).numpy(force=True))
    
    memory_chunks = all_memory_chunks #rename for simplicity
    #memory = np.concatenate(memory_chunks)
    #dicy2 functions
    corpus = GenericCorpus([Dicy2CorpusEvent(content, i, label=label_type([label]))
                                                for (i, (label, content)) in enumerate(corpus_data)],
                                                label_types=[label_type])

    prospector = FactorOracleProspector(corpus, label_type, max_continuity=max_continuity)
    generator = Dicy2Generator(prospector, force_output=force_output)
    print("Corpus (= GT):\n",[label for label,_ in corpus_data])
    print(len(corpus_data))
    return memory_chunks, generator, labels


def generate_response(src_ds : MusicContainer4dicy2, model : Seq2SeqCoupling,
                      chunk_segmentation : str, 
                      with_coupling : bool, k : int, decoding_type : str,
                      generator : Dicy2Generator,
                      temperature : float,
                      entropy_weight : float,
                      batch_size : int,
                      force_coupling : bool,
                      tgt_gts : list[list] = None):

    label_type = ListLabel
    
    collate_fn = MusicDataCollator(unifrom_chunks=chunk_segmentation!="onset")
    src_loader = DataLoader(src_ds,batch_size,False,collate_fn=collate_fn)
    src_fetcher = Fetcher(src_loader)
    src_fetcher.device = model.device
    
    eos = model.special_tokens_idx['eos']
    sos = model.special_tokens_idx["sos"]
    
    queries = [] #slice indexes
    searches_for = [] #classes
    preds = []
    
    # accuracy = []
    
    sliding = src_ds.pre_segmentation_strategy =='sliding'
    
    #compute set of labels in memory for force coupling
    gt_set = None
    if force_coupling:
        gt_set = torch.tensor(list(set(tgt_gts)), device=model.device) #flatten list of gts in memory
        print("GT SET :", gt_set)
    
    if sliding:
        output_hop_size = src_ds.hop_size/src_ds.max_duration #the equivalent hop size in chunks
        
        # hop size manually handled to be an integer. floating point resolution raises an error
        # if not output_hop_size.is_integer():
        #     raise RuntimeError("If hop size is not a multiple of chunk size there wont be an integer number of chunks equivalent to hop_size...")
        
        output_hop_size=int(output_hop_size)
        print(f"sliding window hop duration {src_ds.hop_size}s -> output hop size = {output_hop_size}")
    
    #now we iterate over all source chunks (sub-tracks) to create the whole response to input
    for i in range(len(src_fetcher)):
        src_data = next(src_fetcher)
                
        #generate response labels from source input : "What should be played given what i heard"
        #first encode source
        encoded_src, src_idx, src_pad_mask = model.encode(src_data.src, src_data.src_padding_masks) 
        
        if with_coupling: #if we want to generate a more complex response 
            
            #TODO : ADD ARGUMENT TO USE LAST PREDICTION AS BEGINNING OF TGT ?
            tgt_idx = model.coupling(encoded_src, 
                                       src_pad_mask, 
                                       k, 
                                       max_len=len(encoded_src[0]),
                                       decoding_type=decoding_type,
                                       temperature=temperature,
                                       gt_set=gt_set, #already None if not force coupling
                                       entropy_weight=entropy_weight)[1] #(B,T)
            
            #print("pred :",tgt_idx)
            #print("GT:",tgt_gt)
        
        else : tgt_idx = src_idx #for expermient purposes (identity matching with latent descriptors)
        
        #append predictions for later accuracy calculation
        preds.extend(tgt_idx.reshape(-1).numpy(force=True))
        
        print(tgt_idx)
        
        tgt_idx = tgt_idx[:,1:-1] #remove special_tokens (sos and theoric eos)
        
        
        for j,idxs in enumerate(tgt_idx):
            print("chunk indexes (batch, track segment) :",i,j)
            first = (i==0 and j==0) #first segment
             
            search_for = idxs.numpy(force=True) 
            
            if sliding and not first:
                search_for = search_for[-output_hop_size:] #remove context (only after first segment)

            print(len(search_for),search_for)
            
            #crop response to eos if early stop
            if any(search_for==eos.item()):
                search_for = search_for[:np.where(search_for==eos.item())[0][0]]
            
            #with some models that collapsed there is only the eos token predicted and that riases error by influenceQuery
            if len(search_for)==0 :
                prRed("Silence chunk...")
                if chunk_segmentation=='onset': raise NotImplementedError("Silence handling not implemented for onset segmentation")
                
                silence_q = [-1]*len(src_data.src[j])
                # silence_s = [None]*len(src_data.src[j])
                
                if sliding and not first :
                    silence_q = silence_q[-output_hop_size:]
                    # silence_s = silence_s[-output_hop_size:]
                
                queries.extend(silence_q) #append as many silences as input chunks
                # searches_for.extend(silence_s)
                continue
            
            print(len(search_for),search_for)
            
            searches_for.extend(search_for)
            query = InfluenceQuery([LabelInfluence(label_type([v])) for v in search_for])
            
            t=time.time()
            try :
                output = generator.process_query(query)
            except RecursionError as e:
                
                prRed(f"Recursion Error : {e}")
                print("From query :",query)
                prRed("Generating silent segment")
                output = [None]*len(query)
                
            print(time.time()-t,"s")
            
            #memory slices index to retrieve
            slices=[typing.cast(Dicy2CorpusEvent, v.event).data if v is not None else -1 for v in output]
            print("chunk slices :",slices)        
            
            #add silences to match input length
            min_size = len(src_data.src[j]) if not sliding or first else output_hop_size
            extra_silence = min_size-len(slices)
            
            if extra_silence>0 :
                print(f"adding silence : {extra_silence}")
                if chunk_segmentation=='onset': raise NotImplementedError("Silence handling not implemented for onset segmentation")
                
                silence = [-1]*(extra_silence)
                
                slices.extend(silence) #complete segment slices with silence (-1) to match length
            
            queries.extend(slices)
    
    # accuracy = np.mean(accuracy)
    
    print("response len (in chunks)",len(queries))
    
    return queries, searches_for, preds

def index_to_timestamp(index : int, chunks:np.ndarray):
    if index == -1: 
        t1 = len(np.reshape(chunks,-1))-1
        t0 = t1 - len(chunks[-1])
        return TimeStamp([t0,t1])
    
    t0 = sum([len(chunks[i]) for i in range(index)])
    t1 = t0+len(chunks[index])
    return TimeStamp([t0,t1])

def indexes_to_timestamps(indexes,chunks):
    markers = []
    for index in indexes:
        ts = index_to_timestamp(index,chunks)
        markers.append(ts)
    
    return markers

#function that saves concatenation arguments before concatenation. helps to only reconcatenate by changing a few arguments
def save_and_concatenate(memory_chunks:List, 
                         queries:np.ndarray,
                         concat_fade_time:float,
                         sampling_rate:int,  
                         remove:bool, 
                         max_backtrack:float,
                         save_path : Path) : 
    
    #save params
    np.savez(save_path, 
             chunks = np.asarray(memory_chunks,dtype=object), 
             queries = queries, 
             fade_time = concat_fade_time,
             sampling_rate = sampling_rate,
             remove = remove,
             max_backtrack = max_backtrack)
    
    #concatenate
    response = concatenate_response(memory_chunks, queries, concat_fade_time, sampling_rate, remove, max_backtrack)
    
    return response

def load_and_concatenate(load_path : Path,
                         new_fade_time : Optional[float],
                         new_remove : Optional[bool],
                         new_max_backtrack : Optional[float]) : 
    
    #load data containing arguments for concatenation
    data = np.load(load_path, allow_pickle=True)
    
    #permanent arguments
    memory_chunks = data["chunks"].tolist()
    queries = data["queries"]
    sampling_rate = data["sampling_rate"]
    
    #modifiable arguments
    concat_fade_time = new_fade_time if new_fade_time else data["fade_time"]
    remove = new_remove if new_remove else data["remove"]
    max_backtrack = new_max_backtrack if new_max_backtrack else data["max_backtrack"]
    
    #concatenate
    response = concatenate_response(memory_chunks, queries, concat_fade_time, sampling_rate, remove, max_backtrack)
    
    return response, data

def concatenate_response(memory_chunks:List, 
                         queries:np.ndarray,
                         concat_fade_time:float,
                         sampling_rate:int,  
                         remove:bool, 
                         max_backtrack:float):
    #create concatenate object
    concatenate = Concatenator() 
    
    #construct whole memory from chunks
    memory = np.reshape(memory_chunks,-1)
    
    #extract max_chunk_duration. TODO : WHEN ADDING "new_times" silence handling will be different (tbd)
    max_chunk_duration_samples = len(max(memory_chunks,key=lambda x : len(x)))
    
    #append 2x chunk duration with zeros for silence handling. need 2 times for crossfade purposes
    #silence = np.zeros(int(max_chunk_duration*sampling_rate))
    silence = np.zeros(max_chunk_duration_samples)
    memory_with_silence = np.concatenate([memory,silence,silence]) 
    memory_chunks.extend([silence]*2)
    
    
    #convert queries to timestamps (markers)
    markers = indexes_to_timestamps(queries,memory_chunks)
    
    #create response from queries by concatenating chunks from memory
    response = concatenate(memory_with_silence, markers, sampling_rate,concat_fade_time,remove,max_backtrack)
    
    return response

@torch.no_grad()
def generate(memory_path: Path, src_path:Union[Path,List[Path]], model:Union[Seq2SeqBase,Path],
                      k:int, with_coupling : bool, decoding_type : str, temperature : float, force_coupling : bool,
                      max_track_duration:float,max_chunk_duration:float,
                      track_segmentation:str, chunk_segmentation:str,
                      compute_accuracy : bool,
                      entropy_weight : float = 0,
                      batch_size : int = 8,
                      concat_fade_time : float = 0.04, remove : bool =False, max_backtrack : float = None,
                      device : Optional[torch.device] = None,
                      sampling_rate : int = 16000, tgt_sampling_rates : dict = {'solo':None,'mix':None},
                      max_output_duration : float = None, mix_channels : int = 2, timestamps=[None,None],
                      save_files : bool = True,
                      save_dir : Path = Path('output'),
                      save_concat_args : bool = False,
                      easy_name : bool = False):
    
    if chunk_segmentation=='onset':
        raise ValueError("Concatenation algorithm not compatible with 'onset' segmentation")
    
    if device == None : device = lock_gpu()[0][0]
    
    prYellow("Creating data structure from src and memory paths...")
    #build data structure for model input
    #collate_fn = MusicDataCollator(with_slices=True,unifrom_chunks=chunk_segmentation!="onset")

    memory_ds = MusicContainer4dicy2(memory_path,max_track_duration,max_chunk_duration,sampling_rate,
                                    chunk_segmentation,pre_segemntation=track_segmentation,
                                    timestamps=timestamps[0])


    src_ds =  MusicContainer4dicy2(src_path,max_track_duration,max_chunk_duration,sampling_rate,
                                    chunk_segmentation,pre_segemntation=track_segmentation,
                                    timestamps=timestamps[1])

    #load model if checkpoint path is given
    if isinstance(model,Path):
        prYellow("Loading model from checkpoint...")
        model = load_model_checkpoint(model)
        model.eval()
        model.to(device)
    
    
    prYellow("Generating memory corpus...")
    memory_chunks, generator, labels = generate_memory_corpus(memory_ds,model,chunk_segmentation,batch_size)
    memory = memory_ds.native_track
    
    
    prYellow("Generating reponse...")
    tgt_gts = labels if (force_coupling or compute_accuracy) else None
    queries, searches_for, preds = generate_response(src_ds, model, chunk_segmentation, 
                                                        with_coupling, k, decoding_type, generator, temperature, entropy_weight,
                                                        batch_size, force_coupling, tgt_gts)
    source = src_ds.native_track

    prYellow("Concatenate response...")
    if save_concat_args : 
        concat_file = f"{memory_path.stem}_{max_chunk_duration}s_A{model.codebook_size}.npz"
        
        save_concat_folder = save_dir / "concat_args" #Path(save_dir+f"/concat_args")
        os.makedirs(save_concat_folder, exist_ok=True)
        save_concat_path = save_concat_folder / concat_file
        i=1
        while save_concat_path.exists():
            concat_file = f"{memory_path.stem}_{max_chunk_duration}s_{model.codebook_size}_{i}.npz"
            save_concat_path = save_concat_folder / concat_file
            i+=1
        
        response = save_and_concatenate(memory_chunks,queries,concat_fade_time,memory_ds.native_sr,remove,max_backtrack,save_concat_path)
    
    else : response = concatenate_response(memory_chunks,queries,concat_fade_time,memory_ds.native_sr,remove,max_backtrack)
    
     
    memory = np.array(memory)
    source = np.array(source)
    response = np.array(response)
    
    #normalize to -1,1 and 0 mean
    def normalize(arr):
        norm = np.interp(arr,(arr.min(),arr.max()),(-1,1))
        norm = norm-norm.mean()
        return norm
    
    memory = normalize(memory)
    source = normalize(source)
    response = normalize(response)
    
    query = np.array(queries)
    search_for = np.array(searches_for)
    
    #compute entropy of labels = search for -> diversity
    prYellow("Computing statistics...")
    lengths = compute_consecutive_lengths(query)
    mean_len, median_len, max_len = np.mean(lengths), np.median(lengths), max(lengths)
    
    counts = np.bincount(search_for,minlength=model.codebook_size)
    probs = counts/sum(counts)
    entropy = -sum([p*np.log2(p+1e-9) for p in probs]) #in bits¨
    accuracy = -1
    if compute_accuracy :
        accuracy = compute_acc(np.array(preds), np.array(labels), model.special_tokens_idx['pad'].item())
    
    gt_set=set(labels)
    gt_set_proportion = len(gt_set)/model.vocab_size   
    
    #padding signals
    
    pad = len(response)-len(source)
    if pad > 0 : #response > source
        source = np.concatenate([source,np.zeros(pad)])
    elif pad < 0 : # source > response
        response = np.concatenate([response, np.zeros(abs(pad))])
    
    if mix_channels==2:
        mix = np.concatenate([source[:,None],response[:,None]],axis=1)
    else :
        mix = np.sum([source,response],axis=0)
        mix = normalize(mix) #re-normalize after mean otherwise volume drop
    
    
    #needed when new memory given 
    pad = len(source)-len(memory)
    if pad > 0 : #src > memory
        memory = np.concatenate([memory, np.zeros(abs(pad))])
    elif pad < 0 : # memory > src
        memory = memory[:-abs(pad)] #crop memory to source
    
    
    if mix_channels==2:
        original = np.concatenate([source[:,None],memory[:,None]],axis=1)
    else :
        original = np.sum([source,memory],axis=0)
        original = normalize(original)
        
    
    os.makedirs(save_dir,exist_ok=True)
    
    #wav.write("output.wav",rate=16000,data=response)
    #wav.write("output_mix.wav",rate=16000,data=mix)
    
    if max_output_duration!=None:
        #find non empty in response and crop everything else accordingly
        t0,t1 = find_non_empty(response,max_output_duration,memory_ds.native_sr,return_time=True,find_max=True)
        memory = memory [t0:t1]
        response = response[t0:t1]
        source = source[t0:t1]
        original=original[t0:t1,:] if original.ndim==2 else original[t0:t1]
        mix = mix[t0:t1,:] if mix.ndim ==2 else mix[t0:t1]
    
    #TODO : FIND GOOD WAY TO IMPLEMENT NAMING STRUCTURE OUTSIDE MOISES AND CANONNE
    if save_files:
        prYellow("Saving files...")
        
        #folder_trackname --> moises : 45273_..._voix et cannone : A{i}_Duo2_1_guitare
        if "moises" in str(memory_path):
            track_name = memory_path.parents[1].stem #os.path.basename(os.path.dirname((os.path.dirname(memory_path)))) #track folder
            instrument_name = memory_path.parent.stem #os.path.basename(os.path.dirname(memory_path))
            memory_name = f"{track_name}_{instrument_name}"
        else :
            A_name = memory_path.parent.stem #os.path.basename(os.path.dirname(memory_path))
            memory_name = f"{A_name}_{memory_path.stem}" 
        
        if easy_name : memory_name = "memory" 
        
        memory_folder = None if easy_name else "memory"
        save_file(save_dir,memory_folder,memory_name,memory,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'], make_unique = not easy_name)

        #source_name = directory if moises, track name if canonne
        if "moises" in str(src_path[0]):
            track_name = src_path[0].parents[1].stem #os.path.basename(os.path.dirname((os.path.dirname(src_path[0])))) #track folder
            #instrument_name = os.path.basename(os.path.dirname(src_path[0]))
            source_name = f"{track_name}"
            if len(src_path)==1:
                instrument_name = src_path[0].parent.stem #os.path.basename(os.path.dirname(src_path[0]))
                source_name = f"{track_name}_{instrument_name}"
                
        else :
            if len(src_path)==1:
                A_name = src_path[0].parent.stem #os.path.basename(os.path.dirname(src_path[0]))
                source_name = f"{A_name}_{src_path[0].stem}" #A{i}_{fodler/track_name}
            else : source_name = f"{src_path[0].stem}"  #folder name
        
        if easy_name : source_name = "guide"
        
        source_folder = None if easy_name else "source"
        save_file(save_dir,source_folder,source_name,source,"wav",orig_rate=src_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'], make_unique = not easy_name)
        
        #use same name as memory for response --> crucial for evaluation
        response_name = f"response_{max_chunk_duration}s_A{model.codebook_size}" if easy_name else memory_name 
        response_folder = None if easy_name else "response"
        save_file(save_dir,response_folder,response_name,response,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'], make_unique = not easy_name)
        
        if easy_name :
            mix_name = f"{max_chunk_duration}s_A{model.codebook_size}"
        else : 
            mix_name = f"Cont_{source_name}_Mem_{memory_name}_A{model.codebook_size}_D{max_chunk_duration}_K{k}"
        mix_folder = None if easy_name else "mix"
        save_file(save_dir,mix_folder,mix_name,mix,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['mix'], make_unique = not easy_name)
        
        original_name = "original" if easy_name else source_name
        original_folder = None if easy_name else "original"
        save_file(save_dir,original_folder,original_name,original,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['mix'], make_unique = not easy_name)
        
        save_file(save_dir,"query",f"{mix_name}",query,"txt",orig_rate=None,tgt_rate=None)
        idx = save_file(save_dir,"search_for",f"{mix_name}",search_for,"txt",orig_rate=None,tgt_rate=None)
        
        write_info(model,memory_path, src_path, mix_name, idx, model.name, k, with_coupling, 
                   remove, accuracy, mean_len, median_len, max_len, entropy, w_size=max_chunk_duration,save_dir=save_dir, 
                   decoding_type=decoding_type, force_coupling=force_coupling, temperature=temperature,entropy_weight=entropy_weight,gt_set_portion=gt_set_proportion)
    
    return Munch(memory = memory,
                 source = source,
                 response = response,
                 mix = mix,
                 original = original,
                 search_for = search_for,
                 query = query)

import torchaudio    
def save_file(dir, folder, fname, data, extension, orig_rate, tgt_rate, make_unique=True):
    if folder:
        dir = os.path.join(dir,folder)
    os.makedirs(dir,exist_ok=True)
    idx=0
    path = os.path.join(dir,f"{fname}.{extension}")
    if make_unique:
        while True:
            path = os.path.join(dir,f"{fname}_{idx}.{extension}")
            if not os.path.exists(path):
                break 
            idx+=1
    
    if extension == "wav":
        if tgt_rate != None and orig_rate != tgt_rate:
            #to (c,frames)
            if data.ndim==2:
                data = np.swapaxes(data,0,1) #(c,frames)
            data_tensor = torch.tensor(data)
            
            #resample
            data = torchaudio.functional.resample(data_tensor,orig_rate,tgt_rate).numpy(force=True)
            
            #to (frames,c)
            if data.ndim==2:
                data=np.swapaxes(data,0,1) #(frames,c)
            
            rate=tgt_rate 
        else : rate=orig_rate
        
        sf.write(path,samplerate=rate,data=data.astype(np.float32)) #wavfile expects -1,1 range and float32
        
    elif extension == "txt":
        np.savetxt(path,data,fmt='%d')
    
    return idx

def write_info(model: Seq2SeqBase, memory_path, source_paths, mix_name, idx, model_name, top_k, with_coupling, remove,
               accuracy, mean_len, median_len, max_len, entropy, 
               w_size, save_dir, decoding_type, force_coupling, temperature, entropy_weight,gt_set_portion):
    # Ensure the info directory exists
    info_path = f"{save_dir}/info.txt"
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    
    if not isinstance(source_paths,list): source_paths = [source_paths] 
    
    # Prepare the content to write
    
    content = (
    f"Mix : {mix_name}_{idx} :\n"
    f"\tMemory: {memory_path}\n"
    f"\tSources:\n"
    + "\n".join(f"\t - {path}" for path in source_paths) + "\n"
    f"\tModel : {model_name}\n"
    f"\tParams :\n"
    f"\t\tvocab_size = {model.codebook_size}, segmentation = {model.segmentation}, w_size = {w_size}[s]\n"
    f"\t\ttop-K = {top_k}, with_coupling = {with_coupling}, remove = {remove}, decoding = {decoding_type}, force_coupling = {force_coupling}, temperature = {temperature}, entropy_weight = {entropy_weight}\n"

    f"\tAnalysis :\n"
    f"\t\taccuracy = {accuracy*100:.2f}%, mean_len = {mean_len:.2f}, median_len = {median_len:.2f}, max_len = {max_len}, entropy = {entropy:.2f} [Bits], gt_set_portion = {gt_set_portion*100:.2f}%\n\n"
    )
    
    # Open the file in append mode and write the content
    with open(info_path, 'a') as file:
        file.write(content)
