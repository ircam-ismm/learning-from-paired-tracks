#%%
import os
import shutil
import glob
import soundfile as sf
from librosa import resample, load
from utils.utils import find_non_empty
import numpy as np 
from tqdm import tqdm

#normalize to -1,1
def normalize(arr):
    return np.interp(arr,(arr.min(),arr.max()),(-1,1))

#process function
def process(y,orig_sr,tgt_sr=48000,stereo=True):
    #resample at 48kHz
    y = resample(y=y,orig_sr=orig_sr,target_sr=tgt_sr)
    
    #normalize to -1,1 for waav write format
    y = normalize(y)
    
    #check if stereo
    if stereo:
        if y.ndim != 2:
            y = np.concatenate([y[:,None],y[:,None]],axis=1) #stereo signal
    
    return y


def move_tracks(src_folder,dst_folder,tgt_sr,stereo,basename,max_time=60.):
    t_idx = 1

    p_bar = tqdm(range(len(src_folder)))

    for track in src_folder:
        p_bar.update(1)
        y,orig_sr = load(track,sr=None)
        
        y=find_non_empty(y,max_time,orig_sr)
        
        y = process(y,orig_sr,tgt_sr,stereo)
        
        #needs to be renamed to not overwrite files (they have the same name accross Ai folders)
        dst = os.path.join(dst_folder,f"{basename}{t_idx}.wav")
        
        #save track to folder 
        sf.write(dst,y,samplerate=tgt_sr)
        
        t_idx+=1
    
#%%
canonne_root = '../data/BasesDeDonnees'
canonne_trios = '../data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau'
canonne_duos = '../data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks'
#create isolated tracks folder for audio quality


MAX_TIME = 60. #max 60 seconds of audio to reduce gpu memory during computation
audio_quality_sr = 48000 #we use encodec with stereo files
apa_sr = 16000 #problem with clap so we use PANN expecting 16kHz mono
#%%
#------------ Audio Quality ------------#

#--------------- canonne ---------------#
#find tracks 
trios_val_tracks = glob.glob(canonne_trios+'/val/**/*.wav',recursive=True)
duos_val_tracks = glob.glob(canonne_duos+'/val/**/*.wav',recursive=True)
canonne_val_tracks = trios_val_tracks+duos_val_tracks

trios_test_tracks = glob.glob(canonne_trios+'/test/**/*.wav',recursive=True)
duos_test_tracks = glob.glob(canonne_duos+'/test/**/*.wav',recursive=True)
canonne_test_tracks = trios_test_tracks+duos_test_tracks

#create dst folder
val_quality_folder = os.path.join(canonne_root,'eval/audio_quality/val')
test_quality_folder = os.path.join(canonne_root,'eval/audio_quality/test')

os.makedirs(val_quality_folder,exist_ok=True)
os.makedirs(test_quality_folder,exist_ok=True)

move_tracks(canonne_val_tracks,val_quality_folder,tgt_sr=audio_quality_sr,stereo=True,basename='canonne')
move_tracks(canonne_test_tracks,test_quality_folder,tgt_sr=audio_quality_sr,stereo=True,basename='canonne')
#%%
#--------------moises--------------------#
moises = '../data/moisesdb_v2'

ignore = ["drums.wav",'other.wav','percussion.wav']
moises_val_tracks = glob.glob(moises+'/val/**/*.wav',recursive=True)
moises_val_tracks = [track for track in moises_val_tracks if os.path.basename(track) not in ignore]
moises_test_tracks = glob.glob(moises+'/test/**/*.wav',recursive=True)
moises_test_tracks = [track for track in moises_test_tracks if os.path.basename(track) not in ignore]

#create dst folder
val_quality_folder = os.path.join(moises,'eval/audio_quality/val')
test_quality_folder = os.path.join(moises,'eval/audio_quality/test')

os.makedirs(val_quality_folder,exist_ok=True)
os.makedirs(test_quality_folder,exist_ok=True)

move_tracks(moises_val_tracks,val_quality_folder,audio_quality_sr,True,'moises')
move_tracks(moises_test_tracks,test_quality_folder,audio_quality_sr,True,'moises')    

# %%

#--------------- accompagnement -> APA --------------#
#il nous un dossier avce mix original et un avec mix-1 et random track 

#-------------canonne--------------------#

#val folder


def generate_apa_trios(t_A1,t_A2,t_A3, bg_folder, mis_folder,idx=1):
    p_bar = tqdm(range(len(t_A1)))

    #idx=1

    #Trios
    for A1,A2,A3 in zip(t_A1,t_A2,t_A3):
        p_bar.update(1)
        #open individual tracks
        y1,orig_sr1 = load(A1,sr=None,mono=True)
        y2,orig_sr2 = load(A2,sr=None,mono=True)
        y3,orig_sr3 = load(A3,sr=None,mono=True)
        
        #combine tracks
        bg = np.sum([y1,y2,y3],axis=0) #they should have the same length
        
        #process
        bg = process(bg,orig_sr1,tgt_sr=apa_sr,stereo=False)
        
        #find non empty
        bg = find_non_empty(bg,MAX_TIME,apa_sr)
        
        #pick random track for misaligned
        f = np.random.randint(0,len(As)) #which folder
        t = np.random.choice(As[f]) #which track
        A=[A1,A2,A3] #tracks for bg
        print("original tracks",A)
        print("outside track",t)
        while t in A:
            t = np.random.choice(As[f]) #must be different than original
            print("new outside track",t)
        y_r,orig_sr_r = load(t,sr=None,mono=True)
        
        #pick mix-1
        A = np.array([y1,y2,y3])
        ts = np.random.choice(range(len(A)),size=len(A)-1,replace=False)
        ts=A[ts]
        mix = np.sum(ts,axis=0)
        
        #find non empty segment for mix and distractor
        mix = find_non_empty(mix,MAX_TIME,orig_sr1)
        y_r = find_non_empty(y_r,MAX_TIME,orig_sr_r)
        
        #pad
        pad = len(mix)-len(y_r)
        print("pad",pad)
        if pad > 0 : #mix>y_r
            y_r = np.concatenate([y_r,np.zeros(pad)])
        elif pad < 0 :
            mix = np.concatenate([mix,np.zeros(abs(pad))])
            
        #normalize before mixing for better balance
        y_r = normalize(y_r)
        mix = normalize(mix)
        
        misaligned = np.sum([mix,y_r],axis=0)
        
        #process
        misaligned = process(misaligned,orig_sr_r,tgt_sr=apa_sr,stereo=False)
        
        #non empty 
        #misaligned = find_non_empty(misaligned,MAX_TIME, apa_sr)
        
        #save files
        bg_dst = os.path.join(bg_folder,f"bg{idx}.wav")
        mis_dst = os.path.join(mis_folder,f"misaligned{idx}.wav")
        
        sf.write(bg_dst,bg,samplerate=apa_sr)
        sf.write(mis_dst,misaligned,samplerate=apa_sr)
        
        idx+=1
        
    return idx

def generate_apa_duos(d_A1,d_A2,bg_folder, mis_folder,idx=0):
    p_bar = tqdm(range(len(d_A1)))

    #idx=1
    for A1,A2 in zip(d_A1,d_A2):
        p_bar.update(1)
        #open individual tracks
        y1,orig_sr1 = load(A1,sr=None,mono=True)
        y2,orig_sr2 = load(A2,sr=None,mono=True)
        
        #combine tracks
        bg = np.sum([y1,y2],axis=0) #they should have the same length
        
        #process
        bg = process(bg,orig_sr1,tgt_sr=apa_sr,stereo=False)
        
        bg = find_non_empty(bg,MAX_TIME,apa_sr)
        
        #pick random track for misaligned
        f = np.random.randint(0,len(As)) #which folder
        t = np.random.choice(As[f]) #which track
        A=[A1,A2]
        print("original tracks",A)
        print("outside track",t)
        while t in A:
            t = np.random.choice(As[f]) #must be different than original
            print("new outside track",t)
        y_r,orig_sr_r = load(t,sr=None,mono=True)
        
        #pick mix-1
        A = np.array([y1,y2])
        ts = np.random.choice(range(len(A)),size=len(A)-1,replace=False)
        ts=A[ts]
        mix = np.sum(ts,axis=0)
        
        #find non empty before mixing
        y_r = find_non_empty(y_r,MAX_TIME,orig_sr_r)
        mix = find_non_empty(mix,MAX_TIME,orig_sr1)
        
        #pad
        pad = len(mix)-len(y_r)
        print("pad",pad)
        if pad > 0 : #mix>y_r
            y_r = np.concatenate([y_r,np.zeros(pad)])
        elif pad < 0 :
            mix = np.concatenate([mix,np.zeros(abs(pad))])
            
        #normalize before mixing for better balance
        y_r = normalize(y_r)
        mix = normalize(mix)
        
        misaligned = np.sum([mix,y_r],axis=0)
        
        #process
        misaligned = process(misaligned,orig_sr_r,tgt_sr=apa_sr,stereo=False)
        
        #misaligned = find_non_empty(misaligned,MAX_TIME,apa_sr)
        
        #save files
        bg_dst = os.path.join(bg_folder,f"bg{idx}.wav")
        mis_dst = os.path.join(mis_folder,f"misaligned{idx}.wav")
        
        #sf.write(bg_dst,bg,samplerate=apa_sr)
        #sf.write(mis_dst,misaligned,samplerate=apa_sr)
        
        idx+=1
    return idx
        
canonne_root = '../data/BasesDeDonnees'
val_canonne_trios = '../data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val'
val_canonne_duos = '../data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/val'

d_A1 = val_canonne_duos+'/A1'
d_A1 = [os.path.join(d_A1,track) for track in sorted(os.listdir(d_A1))]

d_A2 = val_canonne_duos+'/A2'
d_A2 = [os.path.join(d_A2,track) for track in sorted(os.listdir(d_A2))]

t_A1 = val_canonne_trios+'/A1'
t_A1 = [os.path.join(t_A1,track) for track in sorted(os.listdir(t_A1))]

t_A2 = val_canonne_trios+'/A2'
t_A2 = [os.path.join(t_A2,track) for track in sorted(os.listdir(t_A2))]

t_A3 = val_canonne_trios+'/A3'
t_A3=[os.path.join(t_A3,track) for track in sorted(os.listdir(t_A3))]

As = [d_A1,d_A2,t_A1,t_A2,t_A3] #all canonne val tracks

val_background_folder = os.path.join(canonne_root,'eval/APA/val/background')
val_misaligned_folder = os.path.join(canonne_root,'eval/APA/val/misaligned')

os.makedirs(val_background_folder,exist_ok=True)
os.makedirs(val_misaligned_folder,exist_ok=True)

idx=generate_apa_trios(t_A1,t_A2,t_A3,val_background_folder,val_misaligned_folder)
generate_apa_duos(d_A1,d_A2,val_background_folder,val_misaligned_folder,idx)
    
    



# %%

#test folder 

test_canonne_trios = '../data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/test'
test_canonne_duos = '../data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/test'

d_A1 = test_canonne_duos+'/A1'
d_A1 = [os.path.join(d_A1,track) for track in sorted(os.listdir(d_A1))]

d_A2 = test_canonne_duos+'/A2'
d_A2 = [os.path.join(d_A2,track) for track in sorted(os.listdir(d_A2))]

t_A1 = test_canonne_trios+'/A1'
t_A1 = [os.path.join(t_A1,track) for track in sorted(os.listdir(t_A1))]

t_A2 = test_canonne_trios+'/A2'
t_A2 = [os.path.join(t_A2,track) for track in sorted(os.listdir(t_A2))]

t_A3 = test_canonne_trios+'/A3'
t_A3=[os.path.join(t_A3,track) for track in sorted(os.listdir(t_A3))]

As = [d_A1,d_A2,t_A1,t_A2,t_A3] #all canonne val tracks

test_background_folder = os.path.join(canonne_root,'eval/APA/test/background')
test_misaligned_folder = os.path.join(canonne_root,'eval/APA/test/misaligned')

os.makedirs(test_background_folder,exist_ok=True)
os.makedirs(test_misaligned_folder,exist_ok=True)

idx=generate_apa_trios(t_A1,t_A2,t_A3,test_background_folder,test_misaligned_folder)
generate_apa_duos(d_A1,d_A2,test_background_folder,test_misaligned_folder,idx)

    
    
# %%

#------------moises------------------#

def generate_apa_moises(moises_folders,distractors,bg_folder,mis_folder,idx=1):
    for folder in moises_folders:
        #remove unwanted instruments
        instruments = glob.glob(folder+'/**/*.wav',recursive=True)
        instruments = [i for i in instruments if os.path.basename(i) not in ignore]
        print("original tracks",instruments)
        #open files 
        ys=[]
        for path in instruments:
            y,orig_sr = load(path,sr=None,mono=True)
            ys.append(y)
        
        lens = [len(y) for y in ys]
        if len(set(lens))!=1:
            max_len = max(lens)
            for i,y in enumerate(ys):
                pad = max_len-len(y)
                if pad>0:
                    ys[i]=np.concatenate([y,np.zeros(pad)])
        ys=np.array(ys)
        
        #combine for background
        bg = np.sum(ys,axis=0)
        
        #process
        bg = process(bg,orig_sr,apa_sr,False)
        
        bg = find_non_empty(bg,MAX_TIME,apa_sr)
        
        #pick mix-1  
        ts = np.random.choice(range(len(ys)),size=len(ys)-1,replace=False)
        print(len(ys),ts)
        mix = np.sum(ys[ts],axis=0)
        
        #pick random
        t = np.random.choice(distractors)
        while t in instruments :
            t = np.random.choice(distractors)
        print("outside track:",t)
        y_r,orig_sr_r = load(t,sr=None,mono=True)
        
        #find non empty segments in distractor and mix
        y_r = find_non_empty(y_r,MAX_TIME,orig_sr_r)
        mix = find_non_empty(mix,MAX_TIME,orig_sr)
        
        #pad
        pad = len(mix)-len(y_r)
        print("pad",pad)
        if pad>0 : #mix>yr
            y_r = np.concatenate([y_r,np.zeros(pad)])
        elif pad<0:
            #pad mix
            mix = np.concatenate([mix,np.zeros(-pad)])
        
        #normalize before summing for better bqlance of distractor and orig
        y_r = normalize(y_r)
        mix=normalize(mix)
        
        #combine
        misaligned = np.sum([mix,y_r],axis=0)
        
        #process
        misaligned=process(misaligned,orig_sr,apa_sr,False)
        
        #misaligned = find_non_empty(misaligned,MAX_TIME,apa_sr)
        
        #save files
        bg_dst = os.path.join(bg_folder,f"bg{idx}.wav")
        mis_dst = os.path.join(mis_folder,f"misaligned{idx}.wav")
        
        sf.write(bg_dst,bg,samplerate=apa_sr)
        sf.write(mis_dst,misaligned,samplerate=apa_sr)
        
        idx+=1
    return idx



moises = '../data/moisesdb_v2'

ignore = ["drums.wav",'other.wav','percussion.wav']
moises_val_folders = glob.glob(moises+'/val/*')



distractors = glob.glob(moises+'/val/**/*.wav',recursive=True)
distractors = [t for t in distractors if os.path.basename(t) not in ignore]

val_background_folder = os.path.join(moises,'eval/APA/val/background')
val_misaligned_folder = os.path.join(moises,'eval/APA/val/misaligned')


os.makedirs(val_background_folder,exist_ok=True)
os.makedirs(val_misaligned_folder,exist_ok=True)

generate_apa_moises(moises_val_folders,distractors,val_background_folder,val_misaligned_folder)





# %%

moises_test_folders = glob.glob(moises+'/test/*')

distractors = glob.glob(moises+'/test/**/*.wav',recursive=True)
distractors = [t for t in distractors if os.path.basename(t) not in ignore]

test_background_folder = os.path.join(moises,'eval/APA/test/background')
test_misaligned_folder = os.path.join(moises,'eval/APA/test/misaligned')

os.makedirs(test_background_folder,exist_ok=True)
os.makedirs(test_misaligned_folder,exist_ok=True)

generate_apa_moises(moises_test_folders,distractors,test_background_folder,test_misaligned_folder)



# %%
