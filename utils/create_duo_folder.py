#!!!!!!!ONLY RUN THIS SCRIPT ONCE !!!!!!
#script to create a pair of folder A1 and A2 containing the individual tracks of the clement cannone duos
import os
import numpy as np
from utils.utils import prGreen
import shutil

root="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks" #now they are in /all

#this folder congtains all separated tracks with a name structure that helps their groupiong in A1 and A2

# name structure Duo{N}_{track_idx}_{instrument}.wav
#group by duo
files = sorted([file for file in os.listdir(root) if file.endswith(".wav")])
duos=[[files[i],files[i+1]] for i in range(0,len(files),2)]
prGreen(f"Duos :{duos}")
A1_files=np.array(duos)[:,0].tolist()
A2_files=np.array(duos)[:,1].tolist()

# create A1,A2 folders if they don't exist
os.makedirs(os.path.join(root, "A1"), exist_ok=True)  
os.makedirs(os.path.join(root, "A2"), exist_ok=True)

A1_paths = [os.path.join(root,file) for file in A1_files]
A2_paths=[os.path.join(root,file) for file in A2_files]

#move files to new folders A1 and A2
dest_A1 = os.path.join(root,"A1")
dest_A2 = os.path.join(root,"A2")

for path1, path2 in zip(A1_paths,A2_paths):  
    #copy the file and move it to A1 and A2 folder
    _ = shutil.copy2(path1,os.path.join(dest_A1,os.path.basename(path1))) 
    _ = shutil.copy2(path2,os.path.join(dest_A2,os.path.basename(path2))) 
