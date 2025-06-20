
#%%
import torch
import numpy as np
from librosa import time_to_frames,frames_to_time
from librosa.onset import onset_backtrack, onset_detect
from typing import Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
#from essentia.standard import Windowing,FFT,CartesianToPolar,FrameGenerator,Onsets,OnsetDetection
#import essentia 

#utilitary functions
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) #function to print in green color in terminal
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) #red
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) #yellow

def depth(L):
    if isinstance(L, list) or isinstance(L,tuple):
        return max(map(depth, L)) + 1
    return 0

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def process_onsets(onsets,min_duration,max_duration): #in smaples or time
    processed_onsets = []
    
    for t0,t1 in zip(onsets[:-1],onsets[1:]):
        if t1-t0 <= max_duration:
            processed_onsets.extend([t0,t1])
        
        else :
            #rechunk too large chunks
            while t1-t0 > max_duration:
                processed_onsets.extend([t0,t0+max_duration])
                t0 = t0+max_duration
            
            processed_onsets.extend([t0,t1]) #append last element less than max_duration
    
    #remove duplicates
    processed_onsets = remove_duplicates(processed_onsets)
    return processed_onsets
                

""" def process_onsets(onsets, min_samples, max_samples):
    # Initialize an empty list to hold the processed segments
    processed_onsets = []
    i=0
    j=1
    while j <= len(onsets) - 1:
        t0,t1 = onsets[i],onsets[j]
        duration=t1-t0
        
        #if in the right range add and next
        if min_samples<=duration<=max_samples:
            processed_onsets.extend([t0,t1])
            i+=1
            j+=1
            continue
        
        #handle big onsets
        while duration > max_samples:
            #chunk by max_duration until duration is less than max
            processed_onsets.extend([t0,t0+max_samples])
            t0 = t0+max_samples
            duration = t1-t0
        
        #if at the end of chunking its more than min add the end of original onset and continue
        if duration>=min_samples:
            processed_onsets.extend([t0,t1])
            i+=1
            j+=1
            continue
        
        elif duration<min_samples and j < len(onsets)-1:
            j=i+1
            #retirer element qui a ete saute
            del onsets[i+1]
    
    processed_onsets = remove_duplicates(processed_onsets)        
    
    return processed_onsets """

def lock_gpu(num_devices=1):
    try :
        import manage_gpus as gpl
        manager=True
    except ModuleNotFoundError as e:
        manager=False
        
    devices=[]
    ids=[]
    
    if manager:
        for i in range(num_devices):
            try:
                    gpu_id_locked = gpl.obtain_lock_id(id=-1)
                    if gpu_id_locked!=-1:
                        device = torch.device(f"cuda:{i}") if num_devices>1 else torch.device("cuda")
                        prYellow(f"Locked GPU with ID {gpu_id_locked} on {device}")
                    else :
                        prRed("No GPU available.")
                        device=torch.device("cpu")
    
            except:
                prRed("Problem locking GPU. Send tensors to cpu")
                device=torch.device("cpu")
                gpu_id_locked=-1
            
            devices.append(device)
            ids.append(gpu_id_locked)
    else :
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        devices = [device]
        ids=[-1]
    
    return devices,ids

def model_params(checkpoint):
    
    ckp = torch.load(checkpoint,map_location=torch.device('cpu'))
    params = ckp['model_params']
    for key, item in params.items():
        prYellow(f"{key} : {item}")
    
    #prYellow(f"\nThe total number of params of the model are {count_model_params(ckp["state_dict"])}")
        
def count_model_params(state_dict : dict):
    count=sum(p.numel() for p in state_dict.values())
    return count


def load_trained_backbone_from_classifier(pretrained_file, backbone_checkpoint="/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt"):
    from architecture.Encoder import Backbone
    from fairseq import checkpoint_utils
    import torch.nn as nn
    
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([backbone_checkpoint])
    baseline_model=models[0]
    
    #w2v backbone adapted for classif
    state_dict=torch.load(pretrained_file,map_location=torch.device("cpu"))
    adapted_model = Backbone(baseline_model,"w2v",mean=True) #adapted model used mean of latents accross time
    
    num_classes=12 #individual instrument labels
    fc = nn.Linear(adapted_model.dim,num_classes)
    
    #list of state dicts if saved whole classifier
    if isinstance(state_dict,list):
        #si .pt contient model
        adapted_model.load_state_dict(state_dict[0])
        fc.load_state_dict(state_dict[1])
        
    
    else: #older trained models only had backbone in state dict
        #si .pt contient state_dict -> instancier model avant de load
        adapted_model.load_state_dict(state_dict)
        prYellow(f"Warning : there is not a checkpoint for the classification head. Returned random classif head.")
    
    classifier = nn.Sequential(
        adapted_model,
        fc
    )
        
    return adapted_model, classifier

def random_sample(topK_idx : torch.Tensor):
    N,k = topK_idx.size()
    random_idx = torch.randint(0,k,size=(N,1),device=topK_idx.device)
    sampled_idx = torch.gather(topK_idx, 1, random_idx).squeeze(1)
    
    return sampled_idx

def sample(topK_probs : torch.Tensor, topK_idx : torch.Tensor):
    #topK_probs (B,k), topK_idx (B,k)
    sampled_idx = torch.multinomial(topK_probs,1) #sample one element from each row in probs, (B,1)
    #print(sampled_idx)
    sampled_idx = topK_idx.gather(1,sampled_idx) #retrieve the corresponding idxs from the sampling of the probs
    #print(topK_probs,sampled_idx)
    return sampled_idx

def topK_search(k, logits : torch.Tensor, targets : Optional[torch.Tensor]=None):
    B,C = logits.size()
    topK_logits,topK_idx = torch.topk(logits,k,dim=-1) #(B,k),(B,k)
    topK_probs = topK_logits.softmax(-1) #apply softmax to create distribution accross topK samples
    if targets != None :
        #find the corresct value in the topK
        #if there isnt then random choice
        sampled_idx=torch.empty(B, dtype=torch.long, device = logits.device) # init empty tenosr
        for i,(topK_sample, topK_prob,tgt) in enumerate(zip(topK_idx, topK_probs, targets)):
            
            idxs = torch.isin(topK_sample,tgt) #is there any occurence of tgt (set_size,) in topK_idxs (K,)
            if any(idxs):
                #sampled_idx[i] = random_sample(topK_sample[idxs].unsqueeze(0)) #if so pick the one with highest probability
                sampled_idx[i] = sample(topK_prob[idxs].unsqueeze(0),topK_sample[idxs].unsqueeze(0))
            
            else :
                #random sample from that list
                #sampled_idx[i] = random_sample(topK_sample.unsqueeze(0)) #fct expects (N,k)-->(1,k)
                sampled_idx[i] = sample(topK_prob.unsqueeze(0),topK_sample.unsqueeze(0))
            
    
    else :
        #return random sample among those values
        #sampled_idx = random_sample(topK_idx)
        
        #return samples following probability distribution
        sampled_idx = sample(topK_probs,topK_idx)
    
    return sampled_idx

def topP_search(p: float, logits : torch.Tensor, targets : Optional[torch.Tensor]=None):
    if not 0<=p<=1 :
        raise ValueError("p value should be between 0 and 1.")
    
    probs = torch.softmax(logits,dim=-1) #(B,vocab)
    sorted_probs, indices = torch.sort(probs,descending=True)
    #print(indices)
    probs_cumsum = torch.cumsum(sorted_probs,dim=-1) #(B,vocab)
    
    close_to_top_p = torch.isclose(probs_cumsum, torch.tensor(p, device=probs_cumsum.device), atol=1e-6)
    
    condition = (probs_cumsum<=p)|close_to_top_p
    
    #if no element satisfies the topP value, return the elem with highest prob
    condition[:,0] = torch.where(condition[:,0]==False,True,condition[:,0])
    #print(condition)
    
    top_p_indices = torch.argwhere(condition)
    #print(top_p_indices)
    
    change = torch.where(torch.diff(top_p_indices[:,0]))[0]+1
    change = torch.cat([torch.tensor([0],device=logits.device),change,torch.tensor([len(top_p_indices)],device=logits.device)])
    
    top_p_indices = [top_p_indices[start:stop,1] for start, stop in zip(change[:-1],change[1:])] #(B,inhomogenous)
    #print(top_p_indices)
    
    #assign topP indices, if no elemnt satisfies p value -> return first
    topP_idx = [indices[i,:idxs[-1]+1] for i,idxs in enumerate(top_p_indices)] #(B,inhomogenous)
    #print("TopP idx:",topP_idx)
    topP_probs = [logits[i][idx].softmax(-1) for i,idx in enumerate(topP_idx)] #(B, inhomogenous)
    #print("TopP probs:",topP_probs)
    
    if targets != None :
        #find the corresct value in the topK
        #if there isnt then random choice
        sampled_idx=torch.empty(len(topP_idx), dtype=torch.long, device = logits.device) # init empty tenosr
        for i,(topP_sample, topP_prob, tgt) in enumerate(zip(topP_idx, topP_probs, targets)):
            
            # if tgt in topP_sample:
            #     sampled_idx[i]=tgt
            idxs = torch.isin(topP_sample,tgt) #is there any occurence of tgt (set_size,) in topP_idxs (>=1,)
            if any(idxs):
                #sampled_idx[i] = random_sample(topP_sample[idxs].unsqueeze(0)) #if so pick the one with highest probability
                sampled_idx[i] = sample(topP_prob[idxs].unsqueeze(0),topP_sample[idxs].unsqueeze(0))
            else :
                #random sample from that list
                #sampled_idx[i] = random_sample(topP_sample.unsqueeze(0)) #fct expects (N,k)-->(1,k)
                sampled_idx[i] = sample(topP_prob.unsqueeze(0),topP_sample.unsqueeze(0))
            
    
    else :
        #return random sample among those values
        #sampled_idx = torch.tensor([random_sample(topP_sample.unsqueeze(0)) for topP_sample in topP_idx], device=logits.device)
        sampled_idx = torch.tensor([sample(topP_prob.unsqueeze(0),topP_sample.unsqueeze(0)) for topP_prob,topP_sample in zip(topP_probs,topP_idx)], device=logits.device)
    
    
    return sampled_idx


def predict_topK_P(k,logits : torch.Tensor, tgt : Optional[torch.Tensor]=None, from_set = False):
    logits_rs = logits.reshape(-1,logits.size(-1)) #(B*T,vocab)
    tgts_rs = None
    if tgt != None:
        if not from_set :
            tgts_rs = tgt.reshape(-1).unsqueeze(1) #(B*T,set size)
        else : 
            #repeat set accross T
            tgt = tgt.unsqueeze(1).repeat(1,logits.size(1),1)
            tgts_rs = tgt.reshape(-1,tgt.shape[-1])
    
    #tgts_rs : (B*T,set size) with set_size = 1 in general and other when force coupling generation
    
    if k>=1:
        preds = topK_search(k,logits_rs,tgts_rs)
    else :
        preds = topP_search(k,logits_rs, tgts_rs)
        
    return preds

def build_coupling_ds(roots : List[List[Path]], 
                      batch_size : int, MAX_TRACK_DURATION, MAX_CHUNK_DURATION,
                    segmentation_strategy="uniform",
                    pre_segmentation='sliding',
                    ignore=[],
                    direction="stem",
                    mask_prob=0.0,
                    mask_len=0,
                    SAMPLING_RATE=16000,
                    distributed=True,
                    device=None):
    
    from MusicDataset.MusicDataset_v2 import MusicCouplingContainer, DataCollatorForCoupling, Fetcher
    from torch.utils.data import DataLoader,DistributedSampler
    import os
    
    #if mask_prob>0 or mask_len> 0 : raise NotImplementedError()

    collate_fn = DataCollatorForCoupling(unifrom_chunks=segmentation_strategy!="onset",sampling_rate=SAMPLING_RATE,mask_prob=mask_prob,mask_len=mask_len)
    
    ds = MusicCouplingContainer(roots, 
                            MAX_TRACK_DURATION, 
                            MAX_CHUNK_DURATION, 
                            SAMPLING_RATE,segmentation_strategy,
                            pre_segmentation=pre_segmentation,
                            ignore_instrument=ignore,
                            direction=direction) #for onset segmentation still needs upgrading to do (handle uneven number of chunks from input and target)
    sampler=None
    shuffle=True
    if distributed:
        sampler=DistributedSampler(ds)
        shuffle=False
    
    loader = DataLoader(ds, batch_size, shuffle=shuffle,sampler=sampler, #with distributed shuffle = false
                        collate_fn=collate_fn,num_workers=2,pin_memory=True)

    fetcher = Fetcher(loader,device)
    
    return fetcher

def build_fine_tuning_ds(guide_path : str, target_path : str, 
                         max_track_duration : float, max_chunk_duration : float, 
                         sampling_rate : int = 16000, segmentation: str = 'uniform', pre_segmentation : str = 'sliding',
                         batch_size : Optional[int] = None, device : torch.cuda.device = None):
    
    from MusicDataset.MusicDataset_v2 import FineTuningDataset, DataCollatorForCoupling, Fetcher
    from torch.utils.data import DataLoader
    
    collate_fn = DataCollatorForCoupling(unifrom_chunks=segmentation!="onset",sampling_rate=sampling_rate)
    
    ds = FineTuningDataset(guide_path, target_path, max_track_duration, max_chunk_duration, sampling_rate,segmentation, pre_segmentation)
    
    if batch_size == None:
        batch_size = len(ds) #batch size is the all track chunks
    
    
    loader = DataLoader(ds, batch_size, shuffle=True,collate_fn=collate_fn)
    
    fetcher = Fetcher(loader, device)
    
    return fetcher

#function to find the memory path in the info.txt file given a mix/response index
def extract_memory_path(file_path, index):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    mix_tag = f"Mix n. {index} :"
    memory_path = None
    
    for i, line in enumerate(lines):
        if mix_tag in line:
            # Look for the Memory line immediately following the mix_tag line
            for j in range(i + 1, len(lines)):
                if "Memory:" in lines[j]:
                    memory_path = lines[j].split("Memory:")[1].strip()
                    return memory_path
    return memory_path

def find_non_empty(track,max_duration,sampling_rate,return_time=False,find_max=False):
        max_samples=int(max_duration*sampling_rate) #convert max_duration in samples
        N=len(track)//max_samples #how many chunks
        if N>0:
            track=track[:N*max_samples] #remove last uneven section
            chunks = track.reshape(-1,max_samples) 
            chunks_norm = (chunks-np.mean(chunks,axis=-1,keepdims=True))/(np.std(chunks,axis=-1,keepdims=True)+1e-5) #Z-score normalize
            energies = np.sum(chunks_norm**2,axis=-1) #energies accross chunks
            #find first occurence of energy above threshold
            if find_max:
                non_empty_chunk_idx = np.argmax(energies)
                
            else : non_empty_chunk_idx = np.where(energies > 0.5)[0][0] if np.any(energies > 0.5) else None
            
            if non_empty_chunk_idx==None : 
                if not return_time:
                    return chunks[0]
                else :
                    return len(chunks[0]),len(chunks[0])+max_samples
            
            non_empty_chunk = chunks[non_empty_chunk_idx]
            if not return_time:
                return non_empty_chunk
            else : return len(chunks[:non_empty_chunk_idx]),len(chunks[:non_empty_chunk_idx])+max_samples
        else : 
            if not return_time:
                return track
            else : return 0,len(track)

def normalize(arr):
    return np.interp(arr,(arr.min(),arr.max()),(-1,1))
    
def compute_consecutive_lengths(idxs : np.ndarray) -> List:
    # if not idxs:
    #     return []
    
    lengths = []
    current_length = 1
    
    for i in range(1, len(idxs)):
        if idxs[i] == idxs[i - 1]+1 and idxs[i-1]!=-1:  # Same segment, increase length
            current_length += 1
        else:  # New segment, save current length and reset
            lengths.append(current_length)
            current_length = 1
    
    # Append the last segment length
    lengths.append(current_length)
    
    return lengths

def compute_identical_idx_lengths(idxs : np.ndarray) -> List:
    # if not idxs:
    #     return []
    
    lengths = []
    current_length = 1
    
    for i in range(1, len(idxs)):
        if idxs[i] == idxs[i - 1]:  # Same segment, increase length
            current_length += 1
        else:  # New segment, save current length and reset
            lengths.append(current_length)
            current_length = 1
    
    # Append the last segment length
    lengths.append(current_length)
    
    return lengths

def broken_histogram_plot(histogram, y_break, title, label, xlabel, ylabel):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,dpi=200)
    fig.subplots_adjust(hspace=0.1)  # adjust space between Axes

    #plot data on same axis
    ax1.bar(range(len(histogram)),histogram,label=label)
    ax2.bar(range(len(histogram)),histogram)

    #zoom in ax2
    ax2.set_ylim(0,y_break)
    #adjust ax1
    ax1.set_ylim(y_break)

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.suptitle(title)
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    #plt.show()
    return fig
    
def broken_histograms_plot(histogram1, histogram2, y_break, title,label_h1,label_h2):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)  # adjust space between Axes

    #plot data on same axis
    ax1.bar(range(len(histogram1)),histogram1,label=label_h1)# align='edge',width=-0.8
    ax1.bar(range(len(histogram2)),histogram2,label=label_h2,alpha = 0.8)
    ax2.bar(range(len(histogram1)),histogram1)
    ax2.bar(range(len(histogram2)),histogram2,alpha = 0.8)

    #zoom in ax2
    ax2.set_ylim(0,y_break)
    #adjust ax1
    ax1.set_ylim(y_break)

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.suptitle(title)
    ax1.grid()
    ax2.grid()
    ax1.legend()
    #plt.show()
    return fig
    
# def detect_onsets(track, sampling_rate, with_backtrack):
#     if sampling_rate<44100 : raise ValueError("The sampling rate for essentia onset detect is otpimized for 44.1kHz. For lower rates use librosa.")
    
#     od_complex = OnsetDetection(method='complex')

#     w = Windowing(type='hann')
#     fft = FFT() # Outputs a complex FFT vector.
#     c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

#     pool = essentia.Pool()
#     for frame in FrameGenerator(track, frameSize=1024, hopSize=512):
#         magnitude, phase = c2p(fft(w(frame)))
#         pool.add('odf.complex', od_complex(magnitude, phase))

#     # 2. Detect onset locations.
#     onsets = Onsets()
#     onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])
    
#     #3 post process onsets : if any onset detected after duration -> remove
#     onsets_complex = onsets_complex[onsets_complex<len(track)/sampling_rate]
    
#     if not with_backtrack:
#         return onsets_complex

#     onsets_backtrack=np.array([])
#     if len(onsets_complex)>0:
#         onset_frames = time_to_frames(onsets_complex,sr=sampling_rate,hop_length=512)

#         onsets_backtrack = onset_backtrack(onset_frames,pool['odf.complex'])
#         onsets_backtrack = frames_to_time(onsets_backtrack,sr=sampling_rate,hop_length=512)
    
#     return onsets_complex, onsets_backtrack

def detect_onsets(audio, sr,with_backtrack):
    onsets = onset_detect(y=audio,sr=sr,backtrack=False,units='time')
    if with_backtrack:
        backtrack = onset_detect(y=audio,sr=sr,backtrack=True,units='time')
        
        return onsets, backtrack
    return onsets

# %%
