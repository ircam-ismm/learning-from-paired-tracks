# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:29:30 2024

@author: balth
"""
#%%
from MusicDataset.MusicDataset_v2 import MusicContainer
from wav2vec2.wav2vec2_utils import DataCollatorForWav2Vec2, Fetcher
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, sys 
import manage_gpus as gpl
import numpy as np
import matplotlib.pyplot as plt
from fairseq import checkpoint_utils
from fairseq.data.data_utils import compute_mask_indices
import argparse
from utils.utils import lock_gpu
from utils.utils import load_trained_backbone_from_classifier
from architecture.Encoder import Backbone

def main():
    
    parser = argparse.ArgumentParser(description="Wav2Vec 2.0 evaluation script.")

    # Add argument for root directory
    parser.add_argument("--root", type=str, required=True,
                        help="Path to the root directory containing the audio data folders to evaluate the model.")
    
    #parser.add_argument("--fname", type=str, default="w2v_bincount.npy",
    #                    help="File name to save the evaluation results (bincount for distribution). Will be saved as a .npy file")
    
    parser.add_argument("--model",type=str, choices=['music','speech','adapted'],required=True)

    args = parser.parse_args()
    
    print(f"Running {__file__} ...")

    #%% Lock GPU

    device=lock_gpu()[0][0]

    #%% load pretrained model adn feature_extractor
    
    from_fairseq=args.model=='music'        
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    if args.model=='speech':
        print("Loading pretrained model from HuggingFace hub")
        model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    elif args.model=='music': 
        checkpoint="../w2v_music_checkpoint.pt"
        print("Loading pretrained model with fairseq package")
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        model=models[0]
    
    elif args.model=='adapted':
        checkpoint="../w2v_music_checkpoint.pt"
        print("Loading pretrained model with fairseq package")
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        pretrained_model=models[0]
        pretrained_file="/data3/anasynth_nonbp/bujard/DICY2/runs/classif_adapt.pt"
        model=Backbone(pretrained_model,type="w2v",mean=False)

    else : raise ValueError()
    
    model=model.to(device)

    #%% Instanciate dataset, datacollator and dataloader
    #dataset arguments
    #music_folders=("Examples", "BasesDeDonnees", "moisesdb_v0.1")
    root = args.root #"/data3/anasynth_nonbp/bujard/data/AudioSet"  #/data3/anasynth_nonbp/bujard/data/"+folder for folder in music_folders]
    max_duration=15.0
    sampling_rate=feature_extractor.sampling_rate
    segmentation_strategy="uniform"

    #dataset containing all the chunks
    eval_ds = MusicContainer(root, max_duration, sampling_rate, segmentation_strategy)

    #DataCollator
    collate_fn = DataCollatorForWav2Vec2(model, feature_extractor, split="eval")

    #dataloader
    batch_size=8
    eval_loader=DataLoader(eval_ds,batch_size,collate_fn=collate_fn,drop_last=True)

    eval_fetcher = Fetcher(eval_loader)
    eval_fetcher.device = device #move fetcher to corresponding device

    #%% Evaluation
    # mean and var update formulas : http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

    # sim = torch.tensor([]) 
    mu, var = 0,0 
    num_bins = 20
    bin_range=(-1.,1.) #cosine similarity range
    bin_count = [0]*num_bins #create 20 bins [-1,1] for histogram
    m = 0 #current number of observations
    n = eval_loader.batch_size #number of new observations (constant defined by batch)
    model.eval()
    model.requires_grad_(False) #no gradient computation for this evaluation
    progress_bar = tqdm(range(len(eval_loader)))
    for i in range(len(eval_loader)):
        inputs=next(eval_fetcher)
        
        with torch.no_grad():
            mask_indices=inputs.mask_indices
            if args.model=='music':
                #using fairseq framework
                #mask_indices=compute_mask_indices(inputs.mask_indices.shape,None, 
                                                #collate_fn.mask_time_prob,collate_fn.mask_time_length)
                #mask_indices=torch.from_numpy(mask_indices)
                outputs=model(source=inputs.x, mask_indices=mask_indices)
                projected_states=outputs['projected_states']
                projected_quantized_states=outputs["projected_quantized_states"]
            elif args.model=='speech' :
                #using HF framework
                #mask_indices=inputs.mask_indices
                outputs = model(inputs.x, mask_time_indices=mask_indices)
                projected_states=outputs.projected_states
                projected_quantized_states=outputs.projected_quantized_states

            elif args.model == 'adapted':
                pass
                        
        cosine_sim=torch.cosine_similarity(projected_states,
                                        projected_quantized_states, dim=-1)
        #print(cosine_sim.shape)
        
        cosine_sim=cosine_sim[mask_indices.to(torch.int)] #only keep masked indices
        #print(cosine_sim.shape)
        
        #update stats
        n_mu = cosine_sim.mean() #mean of current batch
        n_var = cosine_sim.var() #var of current batch
        
        tmp = mu #keep last value of mu for update integrity
        mu = m/(m+n)*tmp + n/(n+m)*n_mu #update mean
        var = m/(m+n)*var + n/(n+m)*n_var + m*n/(m+n)**2 * (tmp-n_mu)**2 #update var
        
        m = m+n #update total number of observations
        
        #update bin count
        new_count = torch.histogram(cosine_sim.flatten().cpu(), bins=num_bins, range=bin_range).hist
        bin_count=[old + new.item() for old, new in zip(bin_count,new_count)]

        if i%max(int(len(eval_loader)*0.1),1)==0:
            #print progress every 10% of total batches or every batch if less that 10 batch
            print("\n",mu.item(), var.item())
            print(bin_count)
        
        progress_bar.update(1)
        
        
        

    print(f"Mean={mu} and std={var**0.5}")
    print(bin_count)

    #save bin_count to npy file for later viz
    fname = f"cosine_sim_{args.model}_{os.path.basename(args.root)}.npy"
    np.save(fname,bin_count)
    #plt.bar(np.linspace(-1,1,num_bins),bin_count)
    #plt.show()


# to use from CLI
if __name__ == "__main__":
    main()




