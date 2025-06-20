import librosa
import numpy as np
from IPython.display import Audio, display
import os, glob
import matplotlib.pyplot as plt
import soundfile as sf

def main():
    #Pour trios
    root = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau"
    folders = ["train","test","val","train_subset","val_subset"]
    folders = [os.path.join(root,folder) for folder in folders]

    for folder in folders:
        files = glob.glob(folder+"/**/*.wav")
        for file in files:
            print("loading file :",file)
            #open file
            y,sr = librosa.load(file,sr=None)
            #normalize
            y_norm = np.interp(y,(y.min(),y.max()),(-1,1)) #(-1,1)
            sf.write(file,data=y_norm.astype(np.float32),samplerate=sr)
            
    root = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks"
    folders = ["train","test","val","train_subset","val_subset"]
    folders = [os.path.join(root,folder) for folder in folders]

    for folder in folders:
        files = glob.glob(folder+"/**/*.wav")
        for file in files:
            print("loading file :",file)
            #open file
            y,sr = librosa.load(file,sr=None)
            #normalize
            y_norm = np.interp(y,(y.min(),y.max()),(-1,1)) #(-1,1)
            sf.write(file,data=y_norm.astype(np.float32),samplerate=sr)

if __name__=="__main__":
    main()