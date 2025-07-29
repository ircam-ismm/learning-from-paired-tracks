from typing import Optional
from pathlib import Path
from utils.dicy2_generator import load_and_concatenate
import numpy as np
import argparse
import soundfile as sf

#TODO : handle naming structure for better retreival of re-cocnat of interest
def re_concatenate(concatenate_load_path : Path,
                    new_fade_time : Optional[float],
                    new_remove : Optional[bool],
                    new_max_backtrack : Optional[float],
                    save_file : bool = True):
    
    
    response, data = load_and_concatenate(concatenate_load_path,
                                    new_fade_time,
                                    new_remove,
                                    new_max_backtrack)
    
    #normalize response to -1,1 and 0 mean
    response = np.interp(response, (response.min(),response.max()),(-1,1)).astype(np.float32)
    response = response - response.mean()
    
    #save file
    if save_file:
        new_fname = f"new_{concatenate_load_path.stem}.wav"
        response_folder = concatenate_load_path.parents[1] / "response" #move up 1 step to be in the directory contaning mix, response etc and go to "response" folder
        file_path = response_folder / new_fname
        idx = 1
        while file_path.exists():
            new_fname = f"new_{concatenate_load_path.stem}_{idx}.wav"
            file_path = response_folder / new_fname
        
        sf.write(file_path,response,data["sampling_rate"])

    return response

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-path", "--concatenate_load_path", type = Path)
    parser.add_argument("--fade_time",type=float)
    parser.add_argument("--remove", action='store_true')
    parser.add_argument("--max_backtrack",type = Optional[float], default = None)
    
    args = parser.parse_args()
    
    re_concatenate(args.concatenate_load_path,
                   args.fade_time,
                   args.remove,
                   args.max_backtrack)