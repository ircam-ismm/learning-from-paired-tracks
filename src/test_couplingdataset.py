#%%
from MusicDataset.MusicDataset_v2 import MusicContainer, MusicContainerPostChunk,MusicDataCollator, MusicContainer4dicy2,MusicCouplingContainer, DataCollatorForCoupling,MusicCouplingDatasetv2
import os
from utils.utils import *
from utils.coupling_ds_generator import extract_all_groups,extract_group 
from torch.utils.data import DataLoader
import numpy as np
from munch import Munch
import torch
import time


#%%
D_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A1"
D_A2="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A2"
T_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A1"
T_A2 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A2"
T_A3 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A3"
train_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]
duos = [[D_A1,D_A2]]
trios = [[T_A1,T_A2,T_A3]]

num_files=50
A1 = [os.path.join(T_A1,t)for t in sorted(os.listdir(T_A1))[:num_files]]
A2 = [os.path.join(T_A2,t)for t in sorted(os.listdir(T_A2))[:num_files]]
A3 = [os.path.join(T_A3,t)for t in sorted(os.listdir(T_A3))[:num_files]]

sub_trio=[[A1,A2,A3]]

#%%
#for moisesdbv2
root="../data/moisesdb_v2/train"
track_folder = os.path.join(root,"63b68795-0076-476b-a917-dec9e89bf91e")
tracks = extract_all_groups(root,instruments_to_ignore=["other","drums","percussion"])
#tracks=[tracks]

#%%

max_chunk_duration=0.5
max_track_duration=5
segmentation_strategy="uniform"
pre_segmentation='sliding'
coupling_ds = MusicCouplingContainer(tracks,max_track_duration, max_chunk_duration,16000,segmentation_strategy,pre_segmentation)#MusicCouplingDataset(root1_train,root2_train,max_track_duration, max_chunk_duration,16000,segmentation_strategy)
#coupling_ds = MusicCouplingDatasetv2(trios,max_track_duration, max_chunk_duration,16000,segmentation_strategy,direction="stem")

t1,t2 = coupling_ds[0]
print(t1.shape,t2.shape)
#%%

#print("C1:",coupling_ds.containers[0].container1.track_chunks[:5])
#print("C1 end:",coupling_ds.containers[0].container1.track_chunks[-5:])
#print("C2:",coupling_ds.containers[0].container2.track_chunks[:5])
#print("C2 end:",coupling_ds.containers[0].container2.track_chunks[-5:])


mask_prob = 0.3
mask_len = 1
num_heads = 4


loader = DataLoader(coupling_ds,8,collate_fn=DataCollatorForCoupling(unifrom_chunks=segmentation_strategy!="onset",
                                                                     mask_prob=mask_prob,mask_len=mask_len),shuffle=True)
load_iter=iter(loader)
t_tot=0
for _ in range(len(loader)):
    t=time.time()
    src, tgt, src_pad_masks, tgt_pad_masks, src_mask_indices = next(load_iter).values()
    print(f"{time.time()-t:.2f}[s]")
    t_tot+=time.time()-t
    print(src.shape,tgt.shape)
    print(src_mask_indices.shape,src_mask_indices)
    print(torch.bincount(src_mask_indices.int().reshape(-1))/len(src_mask_indices.reshape(-1)))
    # T = src.size(1) 
    # src_mask = torch.repeat_interleave(src_mask_indices.unsqueeze(1),repeats=T,dim=1) #(B,T,S)
    # print(src_mask.shape,src_mask)
    # #we need to repeat for every head of each example i.e. example 1 -> head1,head2,...,headN, then example 2 --> repeat on batch dimension
    # src_mask = torch.repeat_interleave(src_mask,repeats = num_heads,dim=0) #(B*heads,T,S)
    # print(src_mask.shape,src_mask)
    #print(src_pad_masks)
    #break
print(f"Total time : {t_tot}")
#prYellow(coupling_ds.inputContainer.audio_chunks)
#prGreen(coupling_ds.targetContainer.audio_chunks)



        
        


# %%
