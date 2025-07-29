#%%
import os
import numpy as np
import shutil
from tqdm import tqdm

np.random.seed(42) #reproductibility but if multiple calls to random will gte different result... so run once
        
#%%
moises = "/data3/anasynth_nonbp/bujard/data/moisesdb_v0.1"
moises_folders = [os.path.join(moises, folder) for folder in os.listdir(moises)]

# Split
# Pour moises on prend simplement 80% des tracks pour train puis 10%/10% val/test
train_folders = np.random.choice(moises_folders, size=int(0.8 * len(moises_folders)), replace=False)
test_val_folders = [folder for folder in moises_folders if folder not in train_folders]
test_folders = test_val_folders[:len(test_val_folders) // 2]
val_folders = test_val_folders[len(test_val_folders) // 2:]

# New directories
moisesv2 = "/data3/anasynth_nonbp/bujard/data/moisesdb_v2"

train_dir = os.path.join(moisesv2, "train")
test_dir = os.path.join(moisesv2, "test")
val_dir = os.path.join(moisesv2, "val")

# Create dirs
os.makedirs(moisesv2, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

#copy paste each folder in their respective directory

#train
bar = tqdm(range(len(train_folders)))
for folder in train_folders:
    src = str(folder)
    dst = os.path.join(train_dir,str(os.path.basename(folder)))
    #print(src)
    #print(dst)
    shutil.copytree(src,dst,dirs_exist_ok=True)
    #break    
    bar.update(1)

#val
bar = tqdm(range(len(val_folders)))
for folder in val_folders:
    src = str(folder)
    dst = os.path.join(val_dir,str(os.path.basename(folder)))
    #print(src)
    #print(dst)
    shutil.copytree(src,dst,dirs_exist_ok=True)
    #break    
    bar.update(1)

#test
bar = tqdm(range(len(test_folders)))
for folder in test_folders:
    src = str(folder)
    dst = os.path.join(test_dir,str(os.path.basename(folder)))
    #print(src)
    #print(dst)
    shutil.copytree(src,dst,dirs_exist_ok=True)
    #break    
    bar.update(1)

#%%
canonne = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/"
duos = os.path.join(canonne,"ClementCannone_Duos/separate_and_csv/separate tracks")

A1 = os.path.join(duos,"A1")
A2 = os.path.join(duos,"A2")

train_dir = os.path.join(duos,"train")
test_dir = os.path.join(duos,"test")
val_dir = os.path.join(duos,"val")

A1_folders = [os.path.join(A1,folder) for folder in sorted(os.listdir(A1))] #need sorted
A2_folders = [os.path.join(A2,folder) for folder in sorted(os.listdir(A2))]

train_idxs = np.random.choice(range(len(A1_folders)),size=int(0.8*len(A1_folders)),replace=False)

A1_train = [A1_folders[idx] for idx in train_idxs]
A1_test_val = [folder for folder in A1_folders if folder not in A1_train]
A1_test = A1_test_val[:len(A1_test_val)//2]
A1_val = A1_test_val[len(A1_test_val)//2:]

A2_train = [A2_folders[idx] for idx in train_idxs]
A2_test_val = [folder for folder in A2_folders if folder not in A2_train]
A2_test = A2_test_val[:len(A2_test_val)//2]
A2_val = A2_test_val[len(A2_test_val)//2:]


#train
dst1 = os.path.join(train_dir,"A1")
dst2 = os.path.join(train_dir,"A2")

os.makedirs(dst1)
os.makedirs(dst2)

for src1,src2 in zip(A1_train,A2_train):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)
    
#test
dst1 = os.path.join(test_dir,"A1")
dst2 = os.path.join(test_dir,"A2")

os.makedirs(dst1)
os.makedirs(dst2)

for src1,src2 in zip(A1_test,A2_test):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)

#val
dst1 = os.path.join(val_dir,"A1")
dst2 = os.path.join(val_dir,"A2")

os.makedirs(dst1)
os.makedirs(dst2)

for src1,src2 in zip(A1_val,A2_val):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)



#%%
canonne = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/"
trios = os.path.join(canonne,"ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau")

#base paths
A1 = os.path.join(trios,"A1")
A2 = os.path.join(trios,"A2")
A3 = os.path.join(trios,"A3")

#splits paths
train_dir = os.path.join(trios,"train")
test_dir = os.path.join(trios,"test")
val_dir = os.path.join(trios,"val")

A1_folders = [os.path.join(A1,folder) for folder in sorted(os.listdir(A1))]
A2_folders = [os.path.join(A2,folder) for folder in sorted(os.listdir(A2))] #need sorted
A3_folders = [os.path.join(A3,folder) for folder in sorted(os.listdir(A3))]

train_idxs = np.random.choice(range(len(A2_folders)),size=int(0.8*len(A2_folders)),replace=False)

A1_train = [A1_folders[idx] for idx in train_idxs]
A1_test_val = [folder for folder in A1_folders if folder not in A1_train]
A1_test = A1_test_val[:len(A1_test_val)//2]
A1_val = A1_test_val[len(A1_test_val)//2:]

A2_train = [A2_folders[idx] for idx in train_idxs]
A2_test_val = [folder for folder in A2_folders if folder not in A2_train]
A2_test = A2_test_val[:len(A2_test_val)//2]
A2_val = A2_test_val[len(A2_test_val)//2:]

A3_train = [A3_folders[idx] for idx in train_idxs]
A3_test_val = [folder for folder in A3_folders if folder not in A3_train]
A3_test = A3_test_val[:len(A3_test_val)//2]
A3_val = A3_test_val[len(A3_test_val)//2:]

#train
dst1 = os.path.join(train_dir,"A1")
dst2 = os.path.join(train_dir,"A2")
dst3 = os.path.join(train_dir,"A3")

os.makedirs(dst1)
os.makedirs(dst2)
os.makedirs(dst3)
print("Train split...")
for src1,src2,src3 in zip(A1_train,A2_train,A3_train):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)
    shutil.copy2(src3,dst3)
    
#test
dst1 = os.path.join(test_dir,"A1")
dst2 = os.path.join(test_dir,"A2")
dst3 = os.path.join(test_dir,"A3")

os.makedirs(dst1)
os.makedirs(dst2)
os.makedirs(dst3)
print("Test split...")
for src1,src2,src3 in zip(A1_test,A2_test,A3_test):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)
    shutil.copy2(src3,dst3)


#val
dst1 = os.path.join(val_dir,"A1")
dst2 = os.path.join(val_dir,"A2")
dst3 = os.path.join(val_dir,"A3")

os.makedirs(dst1)
os.makedirs(dst2)
os.makedirs(dst3)
print("Val split...")
for src1,src2,src3 in zip(A1_val,A2_val,A3_val):    
    shutil.copy2(src1,dst1)
    shutil.copy2(src2,dst2)
    shutil.copy2(src3,dst3)

#pour clement cannone il faut aller directement dans Ai et choisir proportion de tracks
# %%
