
import os
import torch
from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
from architecture.Model import load_model_checkpoint
from MusicDataset.MusicDataset_v2 import Fetcher
from utils.metrics import compute_accuracy,compute_entropy,evaluate_APA,evaluate_audio_quality, evaluate_similarity, compute_consecutive_lengths, compute_WER, compute_LCP_stats
from utils.utils import predict_topK_P,prGreen,build_coupling_ds, extract_memory_path,prYellow, find_non_empty
from utils.coupling_ds_generator import extract_all_groups
from top_k_validity import compute_similarity
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from typing import Union, List, Optional
from librosa import load
from munch import Munch
import csv
import argparse
from generate import generate_example
import datetime
from itertools import permutations


def evaluate_generation(model : Seq2SeqCoupling, eval_fetcher : Fetcher, k: int,  force_coupling : bool,
                        decoding : str = "greedy", temperature : float = 1, entropy_weight : float = 0):
    model.eval()
    acc=0
    diversity = 0 #predicted tokens entropy
    
    accs = []
    acc, acc_std = 0,0
    divs = []
    diversity, diversity_std = 0,0
    perplexs=[]
    perplexity, perplexity_std = 0,0
    wers = []
    wer, wer_std = 0,0
    gt_means = []
    
    preds_, gts = [], [] #for LCP
    
    eos_idx = model.special_tokens_idx['eos'].to(model.device)
    
    runs = 1 #perform multiple runs since stochastic sampling during prediction
    
    with torch.no_grad():
        for n in range(runs):
            print(f"Run #{n+1}")
            for _ in tqdm(range(len(eval_fetcher))):
                inputs = next(eval_fetcher)
                
                
                src, tgt, src_pad_mask, tgt_pad_mask, _ = inputs.values()
                
                gt_tgt_idx = model.encode(tgt,tgt_pad_mask)[1]
                gt_set = torch.unique(gt_tgt_idx) if force_coupling else None
                
                #gt_means.append(len(gt_set)/model.vocab_size)
                
                #generate output
                generated_tgt, generated_tgt_idx, probs = model.generate(src,src_pad_mask,k,
                                                        decoding_type=decoding,
                                                        max_len=gt_tgt_idx.size(1),
                                                        temperature=temperature,
                                                        entropy_weight=entropy_weight,
                                                        gt_set = gt_set)
                
                
                # #check GT stats
                # generated_tgt_idx=gt_tgt_idx.clone().detach()
                # probs = torch.ones(generated_tgt_idx.shape+(model.vocab_size,),device=model.device)
                
                
                #remove <sos>
                tgt_out = gt_tgt_idx[:,1:] 
                generated_tgt_idx = generated_tgt_idx[:,1:]
                probs = probs[:,1:]
                
                preds_.extend(generated_tgt_idx.numpy(force=True))
                gts.extend(tgt_out.numpy(force=True))
                
                # print("-"*10)
                # print(generated_tgt_idx.numpy(force=True))
                # print(tgt_out.numpy(force=True))
                # print(compute_LCP_stats(generated_tgt_idx.numpy(force=True), tgt_out.numpy(force=True)))
                # print("-"*10)
        
                #compute WER
                wers.append(compute_WER(generated_tgt_idx, tgt_out, eos_idx))
                
                #print(probs.shape,generated_tgt_idx.shape)
                
                #compute perplexity of predicted sequence
                preds_probs = probs.reshape(-1,probs.size(-1)).gather(1,generated_tgt_idx.flatten().unsqueeze(1)).squeeze(1) #retrieve logits of predicted tokens (B*T,)
                avg_nll = -sum(torch.log(preds_probs))/len(preds_probs)
                ppl = avg_nll.exp()
                perplexs.append(ppl.item())
                
                preds = generated_tgt_idx.reshape(-1) #(B*T,)
                
                #accuracy with topK
                accs.append(compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=model.special_tokens_idx["pad"]))
                
                #diversity of pred
                divs.append(compute_entropy(preds,min_length=model.vocab_size).item())
                
        
            acc += np.mean(accs)
            acc_std += np.std(accs)
            
            diversity += np.mean(divs)
            diversity_std += np.std(divs)
            
            perplexity += np.mean(perplexs)
            perplexity_std += np.std(perplexs)
            
            wer += np.mean(wers)
            wer_std += np.std(wers)
            
    
    acc, acc_std = acc/runs, acc_std/runs
    diversity, diversity_std = diversity/runs, diversity_std/runs
    perplexity, perplexity_std = perplexity/runs, perplexity_std/runs 
    wer, wer_std = wer/runs, wer_std/runs
    
    # gt_mean = np.mean(gt_means)
    # gt_std = np.std(gt_means)
    
    #pad if necessary for LCP compute
    if len(set([len(p) for p in preds_]))!=1:
        max_len = len(max(preds_, key=lambda x: len(x)))
        preds_ = np.array([np.concatenate([p,-np.ones(max_len-len(p))]) for p in preds_]) #append -1 so its not in the vocabulary
    LCP_stats = compute_LCP_stats(preds_, gts)
    
    return Munch(accuracy={"mean":acc,"std":acc_std},
                 diversity={"mean":diversity,"std":diversity_std},
                 perpleity={"mean":perplexity,"std":perplexity_std},
                 WER = {"mean":wer, "std":wer_std}), LCP_stats, accs, divs, wers, perplexs
    
    

#this function evaluates the Decision module and the perception's quantized encoding
def evaluate_model(model : Seq2SeqBase, eval_fetcher : Fetcher, k: int, device : torch.cuda.device):
    
    model.eval()
    acc=0
    diversity = 0 #predicted tokens entropy
    cosine_sim = 0
    
    accs = []
    divs = []
    sims = []
    cb_usage=[]
    perplexs=[]
    wers = []
    
    eos_idx = model.special_tokens_idx["eos"].to(device)
    codebook = model.vocab_embedding_table
    
    with torch.no_grad():
        for _ in tqdm(range(len(eval_fetcher))):
            inputs = next(eval_fetcher)
            
            if type(model)==Seq2SeqCoupling:
                src, tgt, src_pad_mask, tgt_pad_mask, _ = inputs.values()
                #compute output
                logits, tgt, tgt_idx, _ = model.forward(src, tgt, src_pad_mask, tgt_pad_mask)
                
            elif type(model)==Seq2SeqBase: #for autocompletion
                src, src_pad_mask, _, _ = inputs.values() 
                #compute output
                logits, tgt, tgt_idx, _ = model.forward(src, src_pad_mask)
            
            #encoded inputs
            encoded_src = model.encoder.forward(src, padding_mask = src_pad_mask[0])[1] #only take cb indexes to compute entropy
                    
            tgt_out = tgt_idx[:,1:] #ground truth (B,T) without sos
            
            #topK search
            preds = predict_topK_P(k,logits,tgt_out) #(B*T,)
            
            #compute WER
            wers.append(compute_WER(preds.reshape(logits.shape[:-1]),tgt_out,eos_idx).item())
            
            #compute perplexity of predicted sequence
            probs = logits.softmax(-1)
            preds_probs = probs.reshape(-1,logits.size(-1)).gather(1,preds.unsqueeze(1)).squeeze(1) #retrieve logits of predicted tokens (B*T,)
            avg_nll = -sum(torch.log(preds_probs))/len(preds_probs)
            ppl = avg_nll.exp()
            perplexs.append(ppl.item())
            
            #accuracy with topK
            accs.append(compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=model.special_tokens_idx["pad"]))
            
            #diversity of pred
            divs.append(compute_entropy(preds,min_length=model.vocab_size).item())
            
            #cb usage -> % of vocab size
            cb_usage.append(2**compute_entropy(encoded_src.reshape(-1),min_length=model.codebook_size).item()/model.codebook_size)
            
            #topK validity
            sims.append(compute_similarity(tgt, tgt_idx, logits,
                                             codebook, eos_idx, k).numpy(force=True))
            
            #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    acc = np.mean(accs)
    acc_std = np.std(accs)
    
    diversity = np.mean(divs)
    diversity_std = np.std(divs)
    
    perplexity = np.mean(perplexs)
    perplexity_std = np.std(perplexs)
    
    codebook_usage = np.mean(cb_usage)
    codebook_usage_std = np.std(cb_usage)
    
    sims = np.array(sims) #convert to numpy
    cosine_sim = np.mean(sims[:,0])
    cosine_sim_std = np.mean(sims[:,1]) #mean std over top-K values (dont do another std otherwise its the std of the std)
    
    wer = np.mean(wers)
    wer_std = np.std(wers)
    
    #torch.cuda.empty_cache()
    #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    perfs = Munch(accuracy={"mean":acc,"std":acc_std},
                 diversity={"mean":diversity,"std":diversity_std},
                 perpleity={"mean":perplexity,"std":perplexity_std},
                 codebook_usage = {"mean":codebook_usage,"std":codebook_usage_std},
                 topK_sim = {"mean":cosine_sim,"std":cosine_sim_std},
                 WER = {"mean":wer, "std":wer_std})
    
    return perfs

def generate_eval_examples(tracks_list : List[List[Path]], 
                           model : Seq2SeqCoupling, 
                           k : int, with_coupling : bool, decoding_type : str, temperature : float, force_coupling : bool, 
                           track_duration : float, chunk_duration : float, 
                           segmentation : str, pre_segmentation : str, 
                           crossfade_time : float, max_duration : float, 
                           save_dir : Path, 
                           device : torch.cuda.device,
                           smaller : bool = False, random_subset : bool = True, remove : bool = False, batch_size : int = 8):
    
    #tracks list is like [[t11,t12,...],...,[tm1,tm2,..,tmn]]
    bar = tqdm(range(len(tracks_list)))
    #print("tracks list:",tracks_list)
    for tracks in tracks_list: 
        
        #pick a source and memory
        # memory = np.random.choice(tracks)
        # srcs = [t for t in tracks if t!=memory]
        # print(srcs)
        # if random_subset:
        #     src = np.random.choice(srcs, size = np.random.randint(1,len(srcs)),replace=False).tolist() if len(srcs)>1 else srcs #pick random subset of mix 
        # else : src = srcs
        # print(src)
        
        #print("tracks:",tracks)
        duos = list(permutations(tracks,2))
        #print("duos:",duos)
        
        for memory, src in duos:
            print("Memory :",memory)
            print("Guide :", src)
        
            
            #generate
            generate_example(
                model,
                memory, [src],
                track_duration, chunk_duration, segmentation, pre_segmentation,
                with_coupling, remove, k, decoding_type, temperature, force_coupling,
                crossfade_time, save_dir, smaller, batch_size, True,
                max_duration, device=device, tgt_sampling_rates={'solo':48000,'mix':48000}, mix_channels=1
            )
        
        
        bar.update(1)
        
   
def save_to_file(data: dict, file: Path):
    with open(file, 'a', newline='') as f:  # Open in append mode
        writer = csv.writer(f)
        for key, value in data.items():
            if type(value)==dict:
                writer.writerow([key]+ [(k,v) for k,v in value.items()])  # Append each key-value pair
            else : writer.writerow([key, value])

def parse_args():
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['all','model', 'symbolic_generation','quality','apa','similarity', 'consecutive_idxs','none'],nargs="*")
    parser.add_argument('--model_ckp',nargs='*',type = Path)
    parser.add_argument("--model_ckps_folder",type=Path,default=None)
    parser.add_argument('--data',choices=['canonne','moises']) #canonne/moises
    parser.add_argument("--split", choices = ['val','val_subset','test'])
    parser.add_argument("--batch_size",type=int,default=24)
    parser.add_argument('--k',type=float, default=0.8)
    parser.add_argument("--k_percent_vocab",action="store_true")
    #if already generated audio samples
    parser.add_argument("--generate",action='store_true')
    parser.add_argument("--with_coupling",action='store_true')
    parser.add_argument("--sliding",action='store_true')
    parser.add_argument("--decoding_type",type=str,choices=['greedy','beam'], default='greedy')
    parser.add_argument("--force_coupling",action='store_true')
    parser.add_argument("--temperature",type=float, default=1.)
    parser.add_argument("--entropy_weight",type=float,default=0.)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--max_duration",type=float,default=None)
    parser.add_argument("--crossfade_time",type=float,default=0.05)
    parser.add_argument("--apa_emb",type=str,choices=["CLAP","L-CLAP"], default = "L-CLAP")
    parser.add_argument("--fad_inf",action="store_true")
    #here you can specify "memory" or "original" for baseline evaluation
    parser.add_argument("--quality_tgt_folder",default=None, type = Path, help="target folder ofr generated samples to evaluate audio quality of the model") 
    parser.add_argument("--apa_tgt_folder",default=None, type = Path, help="target folder to generated to evaluate apa")
    parser.add_argument("--similarity_tgt_folder",default=None, type = Path, help="target folder to evaluate music similarity against groudn truth folder. Default is response folder")
    parser.add_argument("--similarity_gt_folder",default=None, type = Path, help="ground truth folder to evaluate music similarity. Default is the memory folder of generated tracks.")
    parser.add_argument("--root_folder", default = None, type = Path, help="root folder path")
    
    args = parser.parse_args()
    
    del parser 
    
    return args

           
def main():
    args = parse_args()
    
    if args.task!=["similarity"]:
        from utils.utils import lock_gpu 
        device = lock_gpu()[0][0]
    else : device = None
    
    
    if args.model_ckp == None:
        assert args.model_ckps_folder != None, "If no model ckp given, specify folder containing all the models to evaluate."
        
        #find all ckp (*.pt) in the folder recursively
        p :Path = args.model_ckps_folder
        model_ckps = sorted(p.glob("**/*.pt")) #sorted(glob.glob(f"{args.model_ckps_folder}/**/*.pt",recursive=True))
        
    else :
        model_ckps : List[Path] = args.model_ckp #with nargs=* --> always as list


    
    
    pre_segmentation = "uniform" if not args.sliding else "sliding"
    
    for model_ckp in model_ckps:
        #prYellow(os.path.basename(model_ckp))
        prYellow(model_ckp.name)        
        
        #extract data
        
        if args.data == 'canonne':
            duos=Path(f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{args.split}")
            duos = [duos.joinpath(A) for A in os.listdir(duos)] #[os.path.join(duos,A) for A in os.listdir(duos)]           
            
            trios = Path(f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{args.split}")
            trios = [trios.joinpath(A) for A in os.listdir(trios)] #[os.path.join(trios,A) for A in os.listdir(trios)]
        
        elif args.data == 'moises':
            moisesdb_val = Path(f"../data/moisesdb_v2/{args.split}")
            moises_tracks = extract_all_groups(moisesdb_val,instruments_to_ignore=['drums','percussion','other'])
        
        if 'all' in args.task:
            args.task = ['model', 'symbolic_generation', 'quality', 'apa', 'similarity']
        
        #load model for generation or model eval
        if 'model' in args.task or 'symbolic_generation' in args.task or args.generate:
            model,params,_ = load_model_checkpoint(model_ckp)
            model.has_masking=False #during evaluation model doesnt mask time indices
            model.to(device)
            
            #extract segmentation params
            track_duration = params['tracks_size']
            chunk_duration = params['chunk_size']
            segmentation_strategy = params['segmentation']
            
            if args.k>=1: 
                k = int(args.k)
            else : 
                k = args.k #total probability
                if args.k_percent_vocab :
                    k = round(args.k*model.vocab_size) #percentage of vocabulary size
        
        
        path=os.path.abspath(__file__)
        dir = Path(os.path.dirname(path))
        eval_dir = dir.joinpath("evaluation")
        
        if args.root_folder != None:
            eval_dir = eval_dir.joinpath(args.root_folder)
        
        k_fname = f"k_{int(args.k)}" if args.k>=1 else f"p_{int(args.k*100)}%"
        
        #TODO : change order for folder structure ?
        save_dir = eval_dir.joinpath(f'{model_ckp.stem}/{args.split}/{args.data}/{k_fname}')
        
        os.makedirs(save_dir,exist_ok=True)
        eval_file = save_dir.joinpath("eval.csv") #save_dir.joinpath("eval.txt")
        
        #depends on dataset and split
        dataset = 'moisesdb_v2' if args.data == 'moises' else 'BasesDeDonnees'
        
        #save arguments/metadata in file
        t=datetime.datetime.now()
        save_to_file({"":"-"*50,'date':t},eval_file)
        
        #generate audio from corresponding data folder
        if args.generate:
            generation_metadata={'task':'generate','k':k,"decoding type":args.decoding_type,"force_coupling":args.force_coupling,"temperature":args.temperature, "entropy_weight":args.entropy_weight, "sliding":args.sliding}
            save_to_file(generation_metadata,eval_file)
            
            prGreen("Generating audio...")
            args.crossfade_time = min(args.crossfade_time,chunk_duration/2) #if fade_t too big for single chunks

            if args.data=='moises':
                generate_eval_examples(moises_tracks,model,k,args.with_coupling,
                                args.decoding_type,args.temperature,args.force_coupling,
                                track_duration,chunk_duration,
                                segmentation_strategy,pre_segmentation,
                                args.crossfade_time,
                                args.max_duration,
                                save_dir,
                                smaller=args.smaller,
                                device=device)
                
            elif args.data == 'canonne':
                #get all tracks in each duo
                dA1_tracks = sorted(duos[0].glob('*.wav')) #sorted(glob.glob(duos[0]+'/*.wav'))
                dA2_tracks = sorted(duos[1].glob('*.wav')) #sorted(glob.glob(duos[1]+'/*.wav'))
                
                duos_tracks = [[t1,t2] for t1,t2 in zip(dA1_tracks,dA2_tracks)]

                generate_eval_examples(
                                        duos_tracks,model,k,args.with_coupling,
                                        args.decoding_type,args.temperature,args.force_coupling,
                                        track_duration,chunk_duration,segmentation_strategy,pre_segmentation,
                                        args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller,
                                        random_subset=False, device=device
                                        )
                
                
                #get all tracks in each trio
                tA1_tracks = sorted(trios[0].glob('*.wav')) #sorted(glob.glob(trios[0]+'/*.wav'))
                tA2_tracks = sorted(trios[1].glob('*.wav')) #sorted(glob.glob(trios[1]+'/*.wav'))
                tA3_tracks = sorted(trios[2].glob('*.wav')) #sorted(glob.glob(trios[2]+'/*.wav'))   
                
                trios_tracks = [[t1,t2,t3] for t1,t2,t3 in zip(tA1_tracks,tA2_tracks,tA3_tracks)]
                
                generate_eval_examples(
                                        trios_tracks,model,k,args.with_coupling,
                                        args.decoding_type,args.temperature,args.force_coupling,
                                        track_duration,chunk_duration,segmentation_strategy,pre_segmentation,
                                        args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller,
                                        random_subset=False,device=device
                                        )
                    
        if 'model' in args.task:
            prGreen("Evaluating model...")
            
            model_eval_metadata = {'task':'model',"k":k, "decoding":"greedy GT selection"}
            save_to_file(model_eval_metadata,eval_file)
            
            #evaliate code related metrics        
            #load dataset 
            if args.data == 'canonne':            
                eval_roots = [duos,trios]
            
            elif args.data == 'moises':
                eval_roots = moises_tracks
            
            eval_fetcher = build_coupling_ds(eval_roots,24,
                                            track_duration,chunk_duration,
                                            segmentation_strategy=segmentation_strategy,
                                            pre_segmentation='uniform',
                                            SAMPLING_RATE=16000,
                                            direction="stem",distributed=False,device=device)
            #eval_fetcher.device = device

            #compute metrics
            output = evaluate_model(model,eval_fetcher,k,device)
            
            #save output to file
            save_to_file(output,eval_file)
            
            #save to npy file
            np.save(save_dir.joinpath(f"model_eval_{t}"), output)
        
        if 'symbolic_generation' in args.task:
            prGreen("Evaluating symbolic generation...")
            
            model_eval_metadata = {'task':'symbolic generation',"k":k, "decoding":args.decoding_type,"force_coupling":args.force_coupling,"temperature":args.temperature}
            save_to_file(model_eval_metadata,eval_file)
            
            #evaliate code related metrics        
            #load dataset 
            if args.data == 'canonne':            
                eval_roots = [duos,trios]
            
            elif args.data == 'moises':
                eval_roots = moises_tracks
            
            eval_fetcher = build_coupling_ds(eval_roots,args.batch_size,
                                            track_duration,chunk_duration,
                                            segmentation_strategy=segmentation_strategy,
                                            pre_segmentation='uniform',
                                            SAMPLING_RATE=16000,
                                            direction="stem",distributed=False,device=device)
            #eval_fetcher.device = device

            #compute metrics
            output,LCP_stats,accs, divs, wers, perplexs = evaluate_generation(model, eval_fetcher, k, args.force_coupling, args.decoding_type, args.temperature, args.entropy_weight)
            
            #save output to file
            #save_to_file({'k':k},eval_file)
            save_to_file(output,eval_file)
            
            #save to npy file
            np.save(save_dir.joinpath(f"symbolic_gen_eval_{t}"), output)
            np.save(save_dir.joinpath(f"LCP_{t}"), LCP_stats)
            np.savez(save_dir.joinpath("symbolic_gen_eval_all"),acc=accs,div=divs,wer=wers,ppl=perplexs)
            
        
        if 'quality' in args.task :
            prGreen("Evaluating audio quality...")
            
            #get the tgt folder if not given
            tgt_folder = args.quality_tgt_folder
            if tgt_folder == None:
                tgt_folder = save_dir.joinpath("response") #os.path.join(save_dir,"response")
            
            #for baseline
            if tgt_folder == "memory":
                tgt_folder = save_dir.joinpath("memory") #os.path.join(save_dir,"memory")
            
            if not tgt_folder.exist() : #os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for audio quality measure. Please generate audio or use '--generate'")
            
            
            quality_metadata = {"task":"audio quality", "fad_inf":args.fad_inf, "tgt folder" : tgt_folder}
            save_to_file(quality_metadata,eval_file)
            
            
                
            ref_folder = f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/audio_quality/{args.split}"
            
            #compute audio quality
            score = evaluate_audio_quality(ref_folder,tgt_folder,args.fad_inf,device=device)
            print("audio quality =",score)
            #save to file
            save_to_file({'audio_quality':score},eval_file)
            
            #save to npy file
            np.save(save_dir.joinpath(f"audio_quality_eval_{t}"), output)
            
        if 'apa' in args.task :
            prGreen("Evaluating APA...")
            
            tgt_folder = args.apa_tgt_folder
            
            if tgt_folder == None:
                #get mix folder 
                tgt_folder = save_dir.joinpath("mix")
            
            #for baseline (higher bound)
            if tgt_folder.name == "original":
                tgt_folder = save_dir.joinpath("original")
            
            #for comparison of apa with only source/guide
            if tgt_folder.name == "source":
                tgt_folder = save_dir.joinpath("source") 
                
            
            if not tgt_folder.exists():
                raise RuntimeError("Wrong or no corresponding folder for APA measure. Please generate audio with '--generate'")
            
            
            apa_metadata = {"task":"APA","embedding":args.apa_emb,"fad_inf":args.fad_inf,"tgt folder":tgt_folder}
            save_to_file(apa_metadata,eval_file)
            
            
            APA_root =  f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/APA/{args.split}"
            bg_folder = APA_root + '/background'
            fake_bg_folder = APA_root+'/misaligned'
            
            #compute APA
            score,fads = evaluate_APA(bg_folder,fake_bg_folder,tgt_folder,args.apa_emb,args.fad_inf,device=device)
            
            print("APA =", score,"\nFADs :",fads)
            #save to file
            save_to_file({'APA':score,"FADs":fads},eval_file)
            
            #save to npy file
            np.save(save_dir.joinpath(f"APA_{args.apa_emb}_{t}"), {'APA':score,"FADs":fads})
        
        if 'similarity' in args.task :
            prGreen("Evaluating Music Similarity...")
            
            tgt_folder = args.similarity_tgt_folder
                        
            #we need to iterate over the response folder
            if tgt_folder == None :
                tgt_folder = save_dir.joinpath("response")
            
            #for baseline
            if tgt_folder.name == "memory":
                tgt_folder=save_dir.joinpath("memory")
                
            print(tgt_folder)
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for similarity measure. Please generate audio with '--generate'")    
            
            
            sim_metadata={"task":"music similarity","tgt folder":tgt_folder}
            save_to_file(sim_metadata,eval_file)
            
            
            targets = sorted(tgt_folder.glob("*.wav")) #sorted(glob.glob(tgt_folder+'/*.wav'))
            
            gt_folder = args.similarity_gt_folder
            if gt_folder==None:
                gt_folder =save_dir.joinpath("memory") #we use the memory folder to compute similarity
            
            if not os.path.exists(gt_folder):
                raise RuntimeError("Wrong or no corresponding ground truth folder for similarity measure.")    
            
            gts = sorted(gt_folder.glob("*.wav"))

            #compute similarity
            sims = []
            for tgt,gt in zip(targets,gts):
                target, tgt_sr= load(tgt,sr=None, mono=True)
                gt, gt_sr = load(gt,sr=None,mono=True)
                
                sim = evaluate_similarity(target, tgt_sr,gt, gt_sr)
                sims.append(sim)
            
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            median_sim = np.median(sims)
            #percentile90 = np.percentile(np.abs(sims),90)
            print(mean_sim,std_sim,median_sim)#,percentile90)
            print(sims)
            #save to file
            save_to_file({"music_similarity (mean, std, median)" : [round(mean_sim,2), round(std_sim,2), round(median_sim,2)]},eval_file)
            
            #save similarities to plot boxes if necessary
            sims_path = save_dir.joinpath(f"music_similarities_{tgt_folder.name}_{t}.npy")
            np.save(sims_path,sims)
        
        #compute consecutive chunk indexes
        if 'consecutive_idxs' in args.task:
            query_folder = save_dir.joinpath("query")
            query_files = query_folder.glob("*.txt")
            
            #TODO : ADD OTHER STATS ?
            #open files, load data
            all_lengths = []
            max_lengths = []
            gt_lengths = []
            for f in query_files:
                idxs = np.loadtxt(f, dtype=int)
                lengths = compute_consecutive_lengths(idxs)
                all_lengths.extend(lengths)
                max_lengths.append(max(lengths))
                gt_lengths.append(len(idxs))
            
            #compute statistics
            lengths_histogram = np.bincount(all_lengths)
            max_lengths_histogram = np.bincount(max_lengths)
            
            #save histogram for plot
            consecutive_idxs_stats_path = save_dir.joinpath(f"consecutive_idxs_stats_{t}.npy")
            np.save(consecutive_idxs_stats_path,{"lengths_histogram":lengths_histogram,"max_lengths_histogram":max_lengths_histogram,"gt_lengths":gt_lengths})
            
            


if __name__=='__main__':
    main()