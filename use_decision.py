from MusicDataset import MusicContainer4dicy2
from utils.dicy2_generator import generate_memory_corpus, generate_response
from architecture.Model import load_model_checkpoint
from utils.utils import lock_gpu
import argparse
from pathlib import Path
from utils.utils import prYellow
import numpy as np
import os 

DEVICES,_=lock_gpu()
DEVICE = DEVICES[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=Path, help = "Path to 'memory' audio file.")
    parser.add_argument('--source', type=Path,nargs="*", help="Path(s) to guiding input audio file(s). Multiple files will automatically be mixed.")
    parser.add_argument("--model_ckp",type=Path, help = "Path to trained model checkpoint. If you choose one of your ptrained models from 'train_model.py', specify the path to the trained VQ on your specific data.")
    parser.add_argument("--VQpath",type=str,default=None,help="path to trained VQ from 'train_model.py' script.")
    parser.add_argument("--with_coupling",action='store_const', default=True, const=False, 
                        help="De-activate coupling, do matching from input to output.") #TODO : change param name
    parser.add_argument("--k",type=float,default=0.8, help = "Top-K/P value. If k<1 use top-P, else top-K. Increasing k/p will add diversity to the output.")
    parser.add_argument("--force_coupling", action = 'store_true', help="Constarined Generation flag.")
    parser.add_argument("--temperature", type = float, default=1., help = "Temperature value. Increasing this parameter will flatten the probability distribution, increasing the diversity of predictions.")
    parser.add_argument("--k_percent_vocab",action="store_true")
    parser.add_argument("-decoding","--decoding_type", type = str, choices=['greedy','beam'], default = "greedy")
    parser.add_argument("--entropy_weight",type=float,default=0.)
    parser.add_argument("--sliding", action = "store_const", default=True, const=False) #TODO:change param name
    parser.add_argument("--save_dir", type=Path, help="Root path to save output files. Defaults to 'output'. Outputs are saved to 'save_dir/{model_ckp}'")
    
    
    
    args = parser.parse_args()
    
    return args

def main(args):
    sampling_rate = 16000
    
    model, params, _ = load_model_checkpoint(args.model_ckp, vq_ckp=args.VQpath)
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
        
        
    #generate memory chunks from Perception
    prYellow("Generating memory corpus...")
    memory_ds = MusicContainer4dicy2(args.memory,track_duration,chunk_duration,sampling_rate,
                                    segmentation,pre_segemntation=pre_segmentation)
    
    batch_size = len(memory_ds)

    memory_chunks, generator, memory_labels = generate_memory_corpus(memory_ds,model,segmentation,batch_size)
    
    
    prYellow("Generating reponse...")
    src_ds =  MusicContainer4dicy2(args.source,track_duration,chunk_duration,sampling_rate,
                                    segmentation,pre_segemntation=pre_segmentation)

    tgt_gts = memory_labels if args.force_coupling else None
    queries, output_labels, preds = generate_response(src_ds, model, segmentation, 
                                                        args.with_coupling, k, args.decoding_type, generator, args.temperature, args.entropy_weight,
                                                        batch_size, args.force_coupling, tgt_gts)
    

    return memory_labels, output_labels

if __name__=="__main__":
    args = parse_args()
    memory_labels, output_labels = main(args)
    
    save_dir = Path("output") if args.save_dir==None else args.save_dir
    save_dir = save_dir.joinpath(args.model_ckp.name).joinpath("use_decision")
    os.makedirs(save_dir,exist_ok=True)
    
    np.save(save_dir.joinpath("memory_labels"),memory_labels,allow_pickle=True)
    np.save(save_dir.joinpath("output_labels"),output_labels,allow_pickle=True)