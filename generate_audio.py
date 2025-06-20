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
from generate import generate_example

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=Path, help = "Path to 'memory' audio file.", required=True)
    parser.add_argument('--source',nargs='*', type=Path, required=True,
                        help="Path(s) to guiding input audio file(s). Multiple files will automatically be mixed.")
    parser.add_argument("--model_ckp",type=Path,nargs="*", required=True,
                        help = "Path(s) to trained model(s) checkpoint(s).")
    parser.add_argument("--with_coupling",action='store_const', default=True, const=False, 
                        help="De-activate coupling, do matching from input to output. TODO : change param name")
    parser.add_argument("--k",type=float,default=0.8, help = "Top-K/P value. If k<1 use top-P, else top-K. Increasing k/p will add diversity to the output.")
    parser.add_argument("--temperature", type = float, default=1., help = "Temperature value. Increasing this parameter will flatten the probability distribution, increasing the diversity of predictions.")
    parser.add_argument("--force_coupling", action = 'store_true', 
                        help="Constarined Generation flag.")
    parser.add_argument("--save_dir", type=Path, help="Root path to save output files. Defaults to 'output'. Outputs are saved to 'save_dir/{model_ckp}'")
    
    #extra arguments for developpement
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--remove",action='store_true')
    parser.add_argument("-decoding","--decoding_type", type = str, choices=['greedy','beam'], default = "greedy")
    parser.add_argument("--entropy_weight",type=float,default=0.)
    parser.add_argument("--k_percent_vocab",action="store_true")
    parser.add_argument('--fade_time',type=float,default=0.05)
    parser.add_argument("--sliding", action = "store_const", default=True, const=False)
    parser.add_argument("--num_examples",type=int, default=1)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--max_duration",type=float, default=None)
    parser.add_argument("--mix_channels",type=int,choices=[1,2],default=2)
    parser.add_argument("-accuracy","--compute_accuracy",action = "store_true")
    parser.add_argument("--save_concat_args", action = "store_true", help = "if True : save concatenation arguments for easier modification of parameters and arguments")
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    
    args = parse_args()
    
    
    from utils.utils import lock_gpu
    DEVICES,_=lock_gpu()
    DEVICE = DEVICES[0]
    
    model_ckps=args.model_ckp
    
    
    for model_ckp in model_ckps:
        print("Generating with model :",model_ckp.name)
        
        save_dir = Path("output") if args.save_dir==None else args.save_dir
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
        
        
        memory = args.memory
        source = args.source   
        
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
            
        