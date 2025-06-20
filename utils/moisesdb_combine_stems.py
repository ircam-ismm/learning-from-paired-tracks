import os
import librosa
import numpy as np
import scipy.io.wavfile as wav

root = "/data3/anasynth_nonbp/bujard/data/moisesdb_v2" #contains train,test,val
splits = [os.path.join(root,split) for split in os.listdir(root)]

ignore = [".DS_Store","data.json"]

for split in splits:
    
    track_folders = [os.path.join(split,track_folder) for track_folder in os.listdir(split)]
    
    for track_folder in track_folders:
        
        instrument_folders = [os.path.join(track_folder,instrument_folder) for instrument_folder in os.listdir(track_folder) if instrument_folder not in ignore] #everything except data.json
        
        for instrument_folder in instrument_folders:
            print(instrument_folder)
            instrument = os.path.basename(instrument_folder)
            
            stems = [os.path.join(instrument_folder,stem) for stem in os.listdir(instrument_folder)]
            
            #open all stems
            ys=[]
            for stem_path in stems:
                sr,y = wav.read(stem_path)
                
                if y.ndim==1: #mono signal -> convert to stereo
                    y = np.stack((y, y), axis=-1)
                ys.append(y)
                
                os.remove(stem_path)
            
            #check that all stems have the same length and pad with 0 if necessary
            max_length = max(y.shape[0] for y in ys)
            ys = [np.pad(y, ((0, max_length - y.shape[0]), (0, 0)), 'constant') for y in ys]
            
            #combine the stems
            combined = np.mean(ys,axis=0)
            
            #save combined track as instrument.wav
            fname = os.path.join(instrument_folder,f"{instrument}.wav")
            wav.write(fname, sr, combined.astype(np.int32))
            
            