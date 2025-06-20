import os
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import List


#root = "../data/moisesdb_v0.1"

def generate_couples(track_folder,instruments_to_ignore=["drums","percussions","other"],with_replacement=True):
    
    instrument_folders = [os.path.join(track_folder,path) for path in os.listdir(track_folder) if not path.endswith(".json")]
    
    #remove unwanted instruments
    if instruments_to_ignore!=None:
        instrument_folders = [folder for folder in instrument_folders if os.path.basename(folder) not in instruments_to_ignore]
    
    #create list of every file
    track_paths = []
    for instrument_folder in instrument_folders:
        for track in os.listdir(instrument_folder):
            track_paths.append(os.path.join(instrument_folder,track))

    fct = combinations_with_replacement if with_replacement else combinations
    
    coupled_tracks = fct(track_paths,2) #create pair of stems from list of all stems in a track
    
    return list(coupled_tracks)

#function to generate list of tuples for coupled stems in every track
#with replacement if we want coupled track with itself
def generate_couples_from_root(root,instruments_to_ignore=None,with_replacement=True):
    
    folders = os.listdir(root)
    all_coupled_tracks = []
    #iterate over all tracks
    for folder in folders:
        track_folder = os.path.join(root,folder)
        
        coupled_tracks = generate_couples(track_folder,instruments_to_ignore,with_replacement) #generate couples from a single track
        
        all_coupled_tracks.extend(coupled_tracks) #add new couples to list of paths
        
def extract_group(track_folder : Path, instruments_to_ignore : List = ["drums","percussions","other"]):
    instrument_folders = [path for path in track_folder.iterdir() if path.suffix != ".json"] #[os.path.join(track_folder,path) for path in os.listdir(track_folder) if not path.endswith(".json")]
    
    #remove unwanted instruments
    instrument_folders = [folder for folder in instrument_folders if folder.name not in instruments_to_ignore]
    
    #create list of every file
    track_paths = []
    for instrument_folder in instrument_folders:
        track_paths.extend(list(instrument_folder.iterdir()))
        # for track in os.listdir(instrument_folder):
        #     track_paths.append(os.path.join(instrument_folder,track))
    
    return track_paths

def extract_all_groups(root : Path, instruments_to_ignore : List = ["drums","percussion","other"]):
    track_folders = list(root.iterdir()) #os.listdir(root)
    all_coupled_tracks = []
    #iterate over all tracks
    for track_folder in track_folders:
        #track_folder = root.joinpath(folder)#os.path.join(root,folder)
        
        coupled_tracks = extract_group(track_folder,instruments_to_ignore) #generate couples from a single track
        
        all_coupled_tracks.append(coupled_tracks) #add new couples to list of paths
    
    return all_coupled_tracks