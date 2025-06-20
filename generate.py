from utils.utils import find_non_empty
from utils.dicy2_generator import generate, load_and_concatenate
from utils.coupling_ds_generator import extract_group
from architecture.Model import load_model_checkpoint
import os
import numpy as np
import argparse
from librosa import load
import glob
from pathlib import Path
from typing import List, Union

def generate_example(model,memory : Path, src : List[Path], track_duration : float, chunk_duration : float, segmentation, pre_segmentation,
                     with_coupling,remove,k, decoding_type, temperature, force_coupling,
                     fade_time,save_dir,smaller, batch_size,
                     compute_accuracy : bool,
                     max_duration=None, 
                     device=None,
                     tgt_sampling_rates : dict = {'solo':None,'mix':None},
                     mix_channels : int = 2,
                     entropy_weight : float = 0,
                     save_concat_args : bool = False,
                     easy_name : bool = False
                     ):
    
    #if device == None : device = lock_gpu[0][0]
    
    if smaller: #find small chunk in memory track
        y,sr = load(memory,sr=None)
        t0,t1 = find_non_empty(y,max_duration,sr,return_time=True)
        timestamps = [[t0/sr,t1/sr],[t0/sr,t1/sr]] #in seconds
    else : timestamps=[None,None]
    
    output = generate(
                    memory,src,model,k,with_coupling,decoding_type, temperature, force_coupling,
                    track_duration,chunk_duration,
                    compute_accuracy=compute_accuracy,
                    entropy_weight=entropy_weight,
                    track_segmentation=pre_segmentation,
                    chunk_segmentation=segmentation,
                    batch_size=batch_size,
                    concat_fade_time=fade_time,
                    remove=remove, timestamps=timestamps,
                    save_dir=save_dir, max_output_duration = max_duration,
                    tgt_sampling_rates=tgt_sampling_rates, mix_channels=mix_channels,
                    device=device,
                    save_concat_args=save_concat_args,
                    easy_name=easy_name
                    )

#Compute accuracy here is true because we use original memory and guide 
def generate_examples(model, chunk_duration, track_duration, segmentation, pre_segmentation,
                      with_coupling, remove, k, decoding_type, temperature, force_coupling, 
                      fade_time, 
                      num_examples, data, save_dir, batch_size, 
                      from_subset=False, 
                      smaller=False, 
                      max_duration=None, 
                      device=None, 
                      mix_channels=2, 
                      entropy_weight : float = 0.,
                      save_concat_args : bool = False,
                      easy_name : bool = False):

    #if device == None : device = lock_gpu[0][0]
    
    val_folder = "val_subset" if from_subset else "val"
    
    if 'canonne' in data:
        #clement cannone
        canonne_t = f"../data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{val_folder}"
        canonne_d = f"../data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{val_folder}"
        
        canonne = [canonne_t,canonne_d]
        if data=='canonne':
            canonne = canonne[np.random.randint(0,2)]
        elif data=='canonne_duos':
            canonne = canonne_d
        elif data=='canonne_trios':
            canonne=canonne_t
        else :
            raise ValueError(f"{data} not good")
        
        A1,A2,A3 = os.path.join(canonne,"A1"),os.path.join(canonne,"A2"),os.path.join(canonne,"A3")
        num_files = len(os.listdir(A1))
        idxs = np.random.choice(range(num_files),size=num_examples,replace=True)

        
        for track_idx in idxs:

            a1 = sorted(os.listdir(A1))[track_idx]
            a2 = sorted(os.listdir(A2))[track_idx]
            
            a1 = os.path.join(A1,a1)
            a2 = os.path.join(A2,a2)
            
            if data=="canonne_trios":
                a3 = sorted(os.listdir(A3))[track_idx]
                a3 = os.path.join(A3,a3)
                tracks = [a1,a2,a3]
            else : tracks = [a1,a2]

            #choose memory and source
            id = np.random.randint(0,len(tracks))
            memory = Path(tracks[id])
            src = [Path(tracks[i]) for i in range(len(tracks)) if i != id ]
            
            generate_example(model, memory, src, 
                             track_duration,chunk_duration,segmentation,pre_segmentation,
                             with_coupling,remove,k,decoding_type,temperature,force_coupling,
                             fade_time,save_dir,
                             smaller,batch_size,
                             max_duration=max_duration,
                             compute_accuracy=True,
                             device=device,
                             mix_channels=mix_channels,
                             entropy_weight=entropy_weight,
                             save_concat_args=save_concat_args,
                             easy_name=easy_name)
        

    elif data == 'moises':
        root= Path(f"../data/moisesdb_v2/{val_folder}")
        track_folders = list(root.iterdir())#[os.path.join(root,track) for track in os.listdir(root)]
        idxs = np.random.choice(range(len(track_folders)),size=num_examples,replace=False)
        for idx in idxs:
            track_folder = track_folders[idx]

            tracks = extract_group(track_folder,instruments_to_ignore=["other","drums","percussion"])
            #print("\n".join(tracks))

            #choose memory and source
            id= np.random.randint(0,len(tracks))
            memory = tracks[id] 
            src = [tracks[i] for i in range(len(tracks)) if i != id ]
            
            generate_example(model, memory, src, 
                             track_duration,chunk_duration,segmentation,pre_segmentation,
                             with_coupling,remove,k,decoding_type,temperature,force_coupling,
                             fade_time,save_dir,smaller,batch_size,max_duration=max_duration,
                             compute_accuracy=True,
                             device=device,
                             entropy_weight=entropy_weight,
                             save_concat_args=save_concat_args,
                             easy_name = easy_name)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=Path, help = "Path to 'memory' audio file.")
    parser.add_argument('--source',nargs='*', type=Path, help="Path(s) to guiding input audio file(s). Multiple files will automatically be mixed.")
    parser.add_argument("--model_ckp",type=Path,nargs="*", help = "Path to trained model checkpoint.")
    parser.add_argument("--with_coupling",action='store_const', default=True, const=False, 
                        help="De-activate coupling, do matching from input to output. TODO : change param name")
    parser.add_argument("--k",type=float,default=0.8, help = "Top-K/P value. If k<1 use top-P, else top-K. Increasing k/p will add diversity to the output.")
    parser.add_argument("--temperature", type = float, default=1., help = "Temperature value. Increasing this parameter will flatten the probability distribution, increasing the diversity of predictions.")
    parser.add_argument("--model_ckps_folder",type=Path)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--remove",action='store_true')
    parser.add_argument("-decoding","--decoding_type", type = str, choices=['greedy','beam'], default = "greedy")
    parser.add_argument("--entropy_weight",type=float,default=0.)
    parser.add_argument("--k_percent_vocab",action="store_true")
    parser.add_argument("--force_coupling", action = 'store_true')
    parser.add_argument('--fade_time',type=float,default=0.05)
    parser.add_argument("--sliding", action = "store_const", default=True, const=False)
    parser.add_argument("--num_examples",type=int, default=1)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--max_duration",type=float, default=None)
    parser.add_argument("--data", choices = ["canonne","canonne_duos","canonne_trios","moises"])
    parser.add_argument("--from_subset", action = 'store_true')
    parser.add_argument("--root_folder",type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--mix_channels",type=int,choices=[1,2],default=2)
    parser.add_argument("-accuracy","--compute_accuracy",action = "store_true")
    parser.add_argument("--save_concat_args", action = "store_true", help = "if True : save concatenation arguments for easier modification of parameters and arguments")
    parser.add_argument("--easy_name", action = 'store_true')
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    
    args = parse_args()
    
    
    from utils.utils import lock_gpu
    DEVICES,_=lock_gpu()
    DEVICE = DEVICES[0]
    
    # If a file with checkpoint paths is provided, read it and add to model_ckp
    if args.model_ckps_folder:
        #find all ckp (*.pt) in the folder recursively
        ckps_folder : Path = args.model_ckps_folder
        model_ckps = sorted(ckps_folder.glob("**/*.pt")) #sorted(glob.glob(f"{args.model_ckps_folder}/**/*.pt",recursive=True))
        
    elif args.model_ckp:
        model_ckps=args.model_ckp
    
    else : raise ValueError("Either give a checkpoint file or a model checkpoint")
    
    for model_ckp in model_ckps:
        print("Generating with model :",model_ckp.name)
        
        save_dir = Path("output") if args.save_dir==None else args.save_dir
        if not args.easy_name :
            save_dir = save_dir.joinpath(model_ckp.name) #os.path.join(save_dir,os.path.basename(model_ckp).split(".pt")[0]) 
        os.makedirs(save_dir,exist_ok=True)
        
        model, params, _ = load_model_checkpoint(model_ckp)
        model.eval()
        _=model.to(DEVICE) 

        chunk_duration=params['chunk_size']
        track_duration=params['tracks_size']
        segmentation = model.segmentation
        pre_segmentation =  "uniform" if not args.sliding else "sliding"
        
        k=args.k
        if k>=1:
            k=int(k)
        elif args.k_percent_vocab:
            k = round(k*model.vocab_size)
        
        #used for generating 'consignes de generation' where there are track folders with "Guide..." and "Mem..." naming structure
        if args.root_folder is not None:
            modes = [mode for mode in list(args.root_folder.iterdir()) if mode.name!=".DS_Store"]
            print(modes)
            for mode_dir in modes:
                save_dir_mode = save_dir.joinpath(mode_dir.name)
                #find all tracks in "relationship mode" folder
                track_folders = list(mode_dir.iterdir())
                pairs=[]
                pairs_exo=[]
                tracks = []
                for track in track_folders:
                    mem = list(track.glob("Mem_*.wav"))
                    guide = list(track.glob("Guide_*.wav"))
                    
                    if len(mem)>0 and len(guide)>0:
                        # tracks.append(track.name)
                        # pairs.append([mem[0], guide[0]])
                        memory = mem[0]
                        source = guide[0]
                        save_dir_orig = save_dir_mode.joinpath(track.name)
                        
                        print(save_dir_orig,track,memory,source)
                        
                        generate_example(model,memory, [source], track_duration, chunk_duration, segmentation, pre_segmentation,
                                    args.with_coupling,args.remove,k,args.decoding_type, args.temperature, args.force_coupling,
                                    args.fade_time,
                                    save_dir_orig,
                                    args.smaller,
                                    args.batch_size,
                                    compute_accuracy=args.compute_accuracy,
                                    max_duration=args.max_duration,
                                    device=DEVICE,
                                    mix_channels=args.mix_channels, 
                                    entropy_weight=args.entropy_weight,
                                    save_concat_args=args.save_concat_args,
                                    easy_name = args.easy_name)
                    
                    mem2 = list(track.glob("Mem2_*.wav"))
                    
                    if len(mem2)>0 and len(guide)>0:
                        memory = mem2[0]
                        source = guide[0]
                        save_dir_exo = save_dir_mode.joinpath(track.name+"_exo")
                        
                        print(save_dir_exo,track,memory,source)
                        
                        generate_example(model,memory, [source], track_duration, chunk_duration, segmentation, pre_segmentation,
                                    args.with_coupling,args.remove,k,args.decoding_type, args.temperature, args.force_coupling,
                                    args.fade_time,
                                    save_dir_exo,
                                    args.smaller,
                                    args.batch_size,
                                    compute_accuracy=args.compute_accuracy,
                                    max_duration=args.max_duration,
                                    device=DEVICE,
                                    mix_channels=args.mix_channels, 
                                    entropy_weight=args.entropy_weight,
                                    save_concat_args=args.save_concat_args,
                                    easy_name = args.easy_name)
                    
                
        
        memory = args.memory
        source = args.source   
        if memory!=None and source!=None:
            generate_example(model,memory, source, track_duration, chunk_duration, segmentation, pre_segmentation,
                            args.with_coupling,args.remove,k,args.decoding_type, args.temperature, args.force_coupling,
                            args.fade_time,
                            save_dir,
                            args.smaller,
                            args.batch_size,
                            compute_accuracy=args.compute_accuracy,
                            max_duration=args.max_duration,
                            device=DEVICE,
                            mix_channels=args.mix_channels, 
                            entropy_weight=args.entropy_weight,
                            save_concat_args=args.save_concat_args,
                            easy_name = args.easy_name)
            
        elif args.data!=None:
            generate_examples(model, chunk_duration, track_duration, segmentation, pre_segmentation,
                                args.with_coupling,args.remove,
                            k, args.decoding_type, args.temperature, args.force_coupling,
                            fade_time=args.fade_time,
                            num_examples=args.num_examples,data=args.data,save_dir=save_dir, 
                            batch_size=args.batch_size,
                            smaller=args.smaller,
                            max_duration=args.max_duration,
                            from_subset=args.from_subset, 
                            device=DEVICE,
                            mix_channels=args.mix_channels,
                            entropy_weight = args.entropy_weight,
                            save_concat_args=args.save_concat_args,
                            easy_name = args.easy_name)
        
            
        #else : raise ValueError("Either specify 'data' or give a source and memory path")