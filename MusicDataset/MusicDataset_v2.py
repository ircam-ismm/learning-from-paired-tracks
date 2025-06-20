# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:44:01 2024

@author: balth
"""

import torch 
from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain
from librosa import load, resample
from librosa.onset import onset_detect
import numpy as np
import audioread, soundfile #for track duration from metadata --> accelerate chunking
from munch import Munch
import os
from tqdm import tqdm
from typing import Union, List, Tuple
from utils.utils import prGreen,prRed,prYellow, remove_duplicates, process_onsets, detect_onsets, normalize
import time
from dataclasses import dataclass
from fairseq.data.data_utils import compute_mask_indices


#list every file in the folder dname
def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['wav','aif','flac']]))
    return fnames

#list of instruments in the moisesdb dataset 
#computed by passing through the directories and creating a set of seen labels
INSTRUMENT_LABELS = ['bass',
 'bowed_strings',
 'drums',
 'guitar',
 'other',
 'other_keys',
 'other_plucked',
 'percussion',
 'piano',
 'vocals',
 'wind',
 'UNK']

INSTRUMENT_TO_LABEL = {label : value for value, label in enumerate(INSTRUMENT_LABELS)}

THRESH = 0.5 #empirical threshold for normalized chunk energy (most of time >0.7 and silent chunks near 0)
MIN_RESOLUTION = 0.025 #sec : minimum sample size for wav2vec models


class MusicContainer(Dataset):
    """
    Dataset class for containing audio chunks used to train and/or evaluate encoding models
    """
    def __init__(self, root:Union[Union[str,Path], List[Union[str,Path]]], max_duration:float, sampling_rate:float, segmentation_strategy:str, 
                 non_empty=False, ignore_instrument = [], max_time:float=600.0, verbose:float=False, init_chunk=True, hop_size : float=None):
        """
        Parameters
        ----------
        root : str or Path
            path to root directory containing audio files.
        max_duration : float
            audio chunks max duration in seconds.
        segmentation_strategy : str
            the segmentation strategy to use for creating the audio chunks.
            only "uniform" segmentation is implemented for now but "mubu" should be implemented
        sampling_rate : int
            sampling rate to load the audio. specified by model configuration and/or
            FeatureExtractor
        from_folder : bool, optional
            if the container is created from a folder of list of paths. The default is True.
        paths : List, optional
            List of audio file paths if from_folder is True. The default is None.

        Raises
        ------
        ValueError
            if from_folder is True and paths is not specified.

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.root=root
        self.max_duration=max_duration
        self.segmentation_strategy = segmentation_strategy
        self.non_empty=non_empty
        self.sampling_rate=sampling_rate
        self.ignore_instrument = ignore_instrument
        self.verbose=verbose
        self.hop_size=self.max_duration*2/3 if not hop_size else hop_size
        
        
        #extract files from folder(s) or paths
        
        self.audio_paths=[]
        #if the root argument is a list of paths
        #look through all of those folders/files
        if isinstance(root, list):
            for folder in root:
                self.audio_paths+=self._find_audio_files(folder, max_time)
        else :
            #root is a single path to a set of folders
            self.audio_paths = self._find_audio_files(root, max_time)
        
        if init_chunk:   
            #segment files into small chunks of max_duration (if uniform)
            fast_algo = segmentation_strategy != 'onset' #apply fast only if uniform or one
            self.audio_chunks=self._create_chunks(self.audio_paths, self.segmentation_strategy, fast=fast_algo)
        
    
    def __len__(self):
        return len(self.audio_chunks)
    
    def __getitem__(self,index):
        #get chunk file, start and end 
        path, start, end, label = self.audio_chunks[index]
        #open file
        duration = end-start #should be max_duration but can change if other than unioform and one segmentation strategy
        
        try:
            chunk, _ = load(path, sr = self.sampling_rate, offset=start, duration=duration, mono=True)
            if duration < self.max_duration: #pad to have uniform chunks when batching
                diff = int((self.max_duration-duration)*self.sampling_rate)
                chunk = np.concatenate([chunk,np.zeros(diff)])
            
            if self.segmentation_strategy in ['one','uniform','sliding'] and self.non_empty:
                #if one chunk per track, be sure it has something in it. NOT TO BE DONE DURING TRAINING
                if not self.check_energy(chunk) : 
                    if self.verbose : prRed("Chunk probably empty. Finding new chunk.")
                    track, _ = load(path, sr = self.sampling_rate, mono=True) #load track once
                    chunk, t0, t1 = self._find_non_empty_chunk_fast(track)
                    #update audio_chunks
                    self.audio_chunks[index]=[path,t0,t1,label]
        
        except Exception as e:
            #print(e) for debugging
            if self.verbose:
                prRed(f"Problem loading chunk from {path} with starting time : {start} [s] and duration : {duration} [s]")
                prYellow("Creating empty chunk of max_duration to handle error")
            chunk = np.zeros(shape=(int(self.sampling_rate*self.max_duration),))
        
        #str label to int
        label = INSTRUMENT_TO_LABEL[label]
        
        data = [chunk, label]
        return data
        
    def check_energy(self,x):
        x_norm =(x-np.mean(x))/(np.std(x)+1e-5)
        E=np.mean(x_norm**2)
        
        if E<THRESH:
            return False
        
        return True
    
    def _get_file_duration(self, path):
        """ 
        This function uses audioread to get the audio duration from the file header
        without loading the entire audio data.
      
        Args:
            filepath: Path to the audio file
      
        Returns:
            float: Audio duration in seconds, or None if duration cannot be determined.
        """
        
        try:
            with audioread.audio_open(path) as f:
                duration = f.duration #file duration in seconds
        except audioread.NoBackendError as e:
            if self.verbose:
                print(f"No compatible backend from audioread for {path}.\nTrying with soundfile.")
            info = soundfile.info(path)
            duration=info.duration
            
            
        
        return duration
    
    def __check_file(self,file_path,max_time=600.0):
        
        label = str(file_path).replace("\\","/").split("/")[-2] 
        if label not in INSTRUMENT_LABELS:
            #check if filename in list (for musdb stem structure)
            fname=os.path.basename(file_path).split(".")[0] #get filename
            if fname in INSTRUMENT_LABELS:
                label = fname
            else : 
                #if the filenmae doesnt give instrument
                label = "UNK" #unknown label (aka instrument)
        #audio path + duration + instrument 
        duration = self._get_file_duration(file_path) #get duration from metadata
        if duration>max_time:  #dont take files too large (long to process)
            return label, None
        
        return label, duration
    
    def _find_audio_files(self, path, max_time=600.0):
        """
        Private function to walk through the given path looking for all the audio files 
        (wav and aif at the moment). This function is responsible for extracting high level metadata
        like the type of audio (instrument) from the folder name or file name
        as the duration of the audio file in seconds.

        Parameters
        ----------
        path : str
            root path to a folder or set of folders containing audio files.

        Returns
        -------
        audio_files : List[(str, float, str)]
            The extracted audio paths with their metadata (instrument and duration).

        """
        audio_files = []
        
        if os.path.isdir(path):
            for root, directories, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.wav', '.aif', '.flac')):  # Consider lowercase extensions
                        # print(root.split("\\")[-1],"\n", file)  
                        file_path = os.path.join(root, file)
                        label, duration = self.__check_file(file_path,max_time)
                        if duration==None or label in self.ignore_instrument:continue #dont take files too large (long to process) or if instrument to ignore
                        audio_files.append([file_path, duration, label])
        
        elif os.path.isfile(path):
            if str(path).lower().endswith(('.wav', '.aif', '.flac')): 
                label, duration = self.__check_file(path,max_time)
                if duration!=None and label not in self.ignore_instrument:
                    audio_files.append([path, duration, label])
            
        return audio_files
    
    def _find_non_empty_chunk(self,track):
        i=0
        max_samples=int(self.max_duration*self.sampling_rate) #[]=[1/s]*[s]
        max_idx=len(track)//max_samples
        chunk=track[i*max_samples:(i+1)*max_samples]
        
        while (not self.check_energy(chunk)) and ((i+1)<max_idx):
            i+=1
            t0=i*max_samples
            t1=(i+1)*max_samples
            chunk=track[t0:t1]
                    
        return chunk, i*self.max_duration, (i+1)*self.max_duration #chunk and t0,t1 for audio_chunks update
    
    def _find_non_empty_chunk_fast(self,track):
        max_samples=int(self.max_duration*self.sampling_rate) #convert max_duration in samples
        N=len(track)//max_samples #how many chunks
        track=track[:N*max_samples] #remove last uneven section
        chunks = track.reshape(-1,max_samples) 
        chunks_norm = (chunks-np.mean(chunks,axis=-1,keepdims=True))/(np.std(chunks,axis=-1,keepdims=True)+1e-5) #Z-score normalize
        energies = np.sum(chunks_norm**2,axis=-1) #energies accross chunks
        #find first occurence of energy above threshold
        non_empty_chunk_idx = np.where(energies > THRESH)[0][0] if np.any(energies > THRESH) else None
        if non_empty_chunk_idx==None:
            if self.verbose:
                prRed("Did not found non-empty chunk. Returning first chunk...")
            return chunks[0],0,self.max_duration
        
        non_empty_chunk = chunks[non_empty_chunk_idx]
        start_time = non_empty_chunk_idx * self.max_duration
        end_time = (non_empty_chunk_idx + 1) * self.max_duration
        return non_empty_chunk, start_time, end_time
        
      
    def segment_track(self, track, strategy="uniform"):
        """

        Parameters
        ----------
        track : np.ndarray(float)
            input track to segment into chunck depending from the strategy argument
        strategy : str, optional
            strategy to use for segmenting the track into chunks.
            "uniform" uses the max_duration attribute to create N chunck of max_samples
            The default is "uniform".

        Returns
        -------
        chunks : List[Tuple(float,float)]
            list of tuple of frame start-end times corresponding to individual chunks
            of the segmented track

        """
        track_duration=len(track)/self.sampling_rate #in seconds
        
        if strategy == "uniform":
            #segment input track into chunks of max_duration seconds
            N = int(track_duration//self.max_duration) #number of max_duration chunks
            r = track_duration%self.max_duration #remainder 
            if N > 0:
                chunks = [[i*self.max_duration,(i+1)*self.max_duration] for i in range(N)]
                
                if r != 0:
                    chunks += [[N*self.max_duration,N*self.max_duration+r]]
                    
            else :
                chunks = [[0, track_duration]]
        
        elif strategy=='one':
            #pick one chunk from whole track
            t0 = np.random.uniform(0,track_duration-self.max_duration)
            chunks = [[t0, t0+self.max_duration]]
        #TODO : MODIFY THIS FUNCTION WITH NEW ONSET_DETCTE
        elif strategy == 'onset':
            #find onset points with backtrack for better segmentation
            onset = onset_detect(y=track,sr=self.sampling_rate,
                                 backtrack=True,units='time',
                                 normalize=True)
            #remove doubles
            onset = remove_duplicates(onset)
            #add 0 and duration for easier chunking
            onset = np.concatenate([[0.],onset,[track_duration]])
            #process onsets
            onset = process_onsets(onset,None,self.max_duration)
            #chunks (in seconds) cropped to max_duration
            #chunks = [[t0,t1] if (t1-t0)<=self.max_duration else [t0,t0+self.max_duration] for t0,t1 in zip(onset[:-1],onset[1:])]
            chunks = [[t0,t1] for t0,t1 in zip(onset[:-1],onset[1:])]
            
        else :
            raise ValueError(f"Invalid segmentation strategy argument {strategy}.")
        
        return chunks
    
    def _segment_track_fast(self,duration, strategy="uniform"):
        #faster segmnetation algorithm using the duration metadata extracted from _find_audio_files and find_duration
        if strategy=="uniform":
            N = int(duration//self.max_duration)
            r = duration%self.max_duration
            
            if N > 0:
                chunks = [[i*self.max_duration,(i+1)*self.max_duration] for i in range(N)]
                
                #Careful could raise error if tracks not of samne length (?)
                if r != 0:
                   chunks += [[N*self.max_duration,N*self.max_duration+r]]
            else :
                chunks = [[0, duration]]
        
        elif strategy=='one':
            #pick one chunk from whole track
            t0 = np.random.uniform(0,duration-self.max_duration)
            chunks = [[t0, t0+self.max_duration]]
        
        elif strategy=="none":
            #no chunking -> useful for MusicCouplingDataset where whole tracks are indexed and then chunked
            chunks=[[0,duration]]
        
        elif strategy=='sliding':
            #hop_size = self.max_duration / 3  # hop size is the third of the chunk size

            chunks = []
            for start in np.arange(0, duration, self.hop_size):
                end = start + self.max_duration
                
                if end > duration : #MAybe xould raise error --> dont take last chunk if greater 
                    end = duration
                    chunk = [start,end]
                    chunks.append(chunk)
                    break
                
                chunk = [start,end]
                
                chunks.append(chunk)
            
        elif strategy == 'onset':
            raise RuntimeError("When applying fast segmentation, onset strategy is not compatible")
            
        else :
            raise ValueError(f"Invalid segmentation strategy argument {strategy}.")
        
        return chunks
    
    def _create_chunks(self, audio_paths, segment_strategy, fast=True):
        
        audio_chunks = []
        
        if self.verbose:
            print("Creating chunks from audio files...")
            progress_bar=tqdm(range(len(audio_paths)))
        
        for path, duration, label in audio_paths:
            if not fast :
                #open file --> really slow, use the metadata from audioread if simple segment strategy
                track, _ = load(path, sr = self.sampling_rate, mono=True)
                #get chunks of track
                chunks = self.segment_track(track, strategy=segment_strategy)
            else :
                chunks = self._segment_track_fast(duration, strategy=segment_strategy)
            
            #append path to the chunks
            chunks_with_path = [[path, start, end, label] for start, end in chunks]
            
            #append list to set of total audio_chunks
            audio_chunks += chunks_with_path
            
            if self.verbose:
                progress_bar.update(1)
            # progress_bar.write(f"\nFinished segmenting track {path}")
        
        if self.verbose : 
            print("Chunks created and stored.")
        
        return audio_chunks
    
    def get_class_distribution(self, idx=None):
        #add idx arg because in some cases torch.random_split is used and retursn a torch.Subset (whole Datasset, indices)
        bin_counts=np.zeros(len(INSTRUMENT_LABELS))
        if idx is None :
            idx=range(len(self.audio_paths))
        
        print("Computing distribution...")
        progress_bar=tqdm(idx)
        for i in idx:
            label=self[i][-1]
            bin_counts[label]+=1
            progress_bar.update(1)
            
        distribution=bin_counts/sum(bin_counts)
        
        return distribution 

#special container that does chunking per track and not on whole ds
class MusicContainerPostChunk(MusicContainer):
    def __init__(self, root:Union[Union[str,Path], List[Union[str,Path]]], track_duration:float, max_duration:float, sampling_rate:float, 
                 segmentation_strategy:str, pre_segmentation = 'sliding', non_empty=False, ignore_instrument=[], max_time:float=600.0, verbose:float=False):
        """
        Parameters
        ----------
        root : str or Path
            path to root directory containing audio files.
        max_duration : float
            audio chunks max duration in seconds.
        segmentation_strategy : str
            the segmentation strategy to use for creating the audio chunks.
            only "uniform" segmentation is implemented for now but "mubu" should be implemented
        sampling_rate : int
            sampling rate to load the audio. specified by model configuration and/or
            FeatureExtractor
        from_folder : bool, optional
            if the container is created from a folder of list of paths. The default is True.
        paths : List, optional
            List of audio file paths if from_folder is True. The default is None.

        Raises
        ------
        ValueError
            if from_folder is True and paths is not specified.

        Returns
        -------
        None.

        """
        
        #super init max duration is track duration
        super().__init__(root, track_duration, sampling_rate, segmentation_strategy,
                 non_empty,ignore_instrument, max_time, verbose, init_chunk=False)
        
        
        #create track chunks
        
        self.track_chunks = self._create_chunks(self.audio_paths,pre_segmentation,fast=True) 
        self.max_duration=max_duration #after first unifrom chunking of tracks reassing max_duration to chunk max_duration for later segmentation
        self.track_duration=track_duration
    
    def __len__(self):
        return len(self.track_chunks)
    
    def __getitem__(self, index):
        #get track
        path,start,end,label = self.track_chunks[index]
        duration=end-start
        track, _= load(path,sr=self.sampling_rate,offset=start,duration=duration,mono=True)    
        
        #pad with zeros if track is not track_duration --> raises error in batching
        if duration<self.track_duration:
            pad = int((self.track_duration-duration)*self.sampling_rate)
            track = np.concatenate([track,np.zeros(pad)])
        
        #segment track
        chunks = self.segment_track(track,self.segmentation_strategy)
        
        label = INSTRUMENT_TO_LABEL[label]
        
        return chunks, label
    
    #overwrite segment track on samples and not time
    def segment_track(self,track,strategy,return_slices=False):
        if strategy == 'uniform':
            chunks = self.segment_track_uniform(track,return_slices)
        
        elif strategy == 'onset':
            chunks = self.segment_track_onset(track,return_slices)
        
        elif strategy == "sliding":
            chunks = self.segment_track_sliding(track,return_slices)
        
        else :
            raise ValueError(f"Wrong argument value for 'strategy' : {strategy}")
        
        return chunks
    
    #new segmentation methods for correct track chunking
    def segment_track_uniform(self,track,return_slices=False):
        max_samples = int(self.max_duration*self.sampling_rate) #max samples per chunkn. CAREFUL IF MAX*SR IS NOT AN INT !!!!!
        r = len(track)%max_samples #remaining samples
        pad_len = int(max_samples-r) if r>0 else 0 #padding length to have equal sized chunks
        pad = np.zeros(pad_len)
        track_padded = np.concatenate([track,pad])
        chunks = track_padded.reshape(-1,max_samples)
        
        if return_slices:
            slices = np.array([i*max_samples,(i+1)*max_samples] for i in range(len(chunks)))
            return chunks,slices
        
        return chunks

    def segment_track_onset(self, track, return_slices=False):
        #find onset points with backtrack for better segmentation
        #onset = onset_detect(y=track,sr=self.sampling_rate,backtrack=True,normalize=True,units='samples')
        y = resample(track,orig_sr=self.sampling_rate,target_sr=44100) #resample for essentia onset segmentation (optimized for 44.1khz)
        onset = detect_onsets(y,44100,with_backtrack=True)[1] #in seconds
        onset = (onset*self.sampling_rate).astype(int) #in samples of original tracks
        
        #remove doubles
        onset = remove_duplicates(onset)
        
        #add 0 and max_samples for easier chunking
        onset = np.concatenate([[0],onset, [len(track)]])
        
        #handle too long onsets. min is given by resolution of wav2vec but not used
        onset = process_onsets(onset,min_duration=int(MIN_RESOLUTION*self.sampling_rate),max_duration=int(self.max_duration*self.sampling_rate))
        #chunks 
        chunks = [track[int(t0):int(t1)] for t0,t1 in zip(onset[:-1],onset[1:])] #returned as list cuz inhomogenous
        
        #pad chunks to max_len
        #max_len = len(max(chunks,key=lambda x : len(x)))
        
        #pad to equal size chunks and stack
        #chunks = np.vstack([np.concatenate([chunk,np.zeros(max_len-len(chunk))]) for chunk in chunks])
        
        if return_slices:
            slices = [[t0,t1] for t0,t1 in zip(onset[:-1],onset[1:])]
            return chunks, slices

        return chunks
    
    def segment_track_sliding(self, track, return_slices=False):
        max_samples = int(self.max_duration * self.sampling_rate)  # max samples per chunk
        hop_size_samples = int(self.hop_size*self.sampling_rate) #max_samples // 2  # hop size is half of the chunk size

        chunks = []
        slices = []
        for start in range(0, len(track), hop_size_samples):
            end = start + max_samples
            if end > len(track):
                # If the end exceeds the track length, pad the last chunk
                pad_len = end - len(track)
                chunk = np.concatenate([track[start:], np.zeros(pad_len)])
            else:
                chunk = track[start:end]
            chunks.append(chunk)
            
            if return_slices:
                slices.append([start,end])
        
        # Convert list of chunks to numpy array
        chunks = np.array(chunks)
        
        if return_slices:
            slices = np.array(slices)
            return chunks, slices

        return chunks

#TODO : make this class more pythonic : inherit from conainer and maybe modify container
class MusicContainer4dicy2(Dataset):
    def __init__(self,track_path:Union[Union[str,Path], List[Union[str,Path]]], 
                 track_duration:float, max_duration:float, sampling_rate:float, 
                 segmentation_strategy:str, pre_segemntation:str='uniform',
                 timestamps=None, hop_fraction:float=0.70):
        
        super().__init__()
        self.sampling_rate=sampling_rate
        self.track_duration=track_duration
        self.max_duration=max_duration
        self.segmentation_strategy=segmentation_strategy
        self.pre_segmentation_strategy = pre_segemntation
        if timestamps!=None:
            t0,t1=timestamps 
            duration = t1-t0
        else :
            t0=0
            duration = None
        
        #hop size has to be a multiple (N) of chunk size for sliding generation
        N = int(hop_fraction*self.track_duration/self.max_duration)
        self.hop_size = N*self.max_duration #self.track_duration*2/3 if not hop_size else hop_size
        
        if isinstance(track_path,List): #for guide
            #open each track and combine
            tracks=[]
            ls=[]
            native_tracks = []
            ls_nat=[]
            for path in track_path:
                track,_=load(path,sr=sampling_rate,mono=True, offset=t0,duration=duration)
                native_track, native_sr = load(path,sr=None,mono=True,offset=t0,duration=duration)
                
                ls.append(len(track))
                ls_nat.append(len(native_track))
                
                tracks.append(track)
                native_tracks.append(native_track)
            
            #pading (shouldnt be necessary but for moises it raised an error at some point)
            if len(set(ls))!=1:
                l_max = max(ls)
                l_max_nat=max(ls_nat)
                for i in range(len(tracks)):
                    pad = l_max-ls[i]
                    pad_nat = l_max_nat-ls_nat[i]
                    if pad!=0: #true for native and resampled
                        tracks[i] = np.concatenate([tracks[i],np.zeros(pad)]) 
                        native_tracks[i]=np.concatenate([native_tracks[i],np.zeros(pad_nat)]) 
            
            track = np.sum(tracks,axis=0) #combined tracks
            track = np.interp(track,(track.min(),track.max()),(-1,1))
            native_track = np.sum(native_tracks,axis=0) 
            native_track = np.interp(native_track,(native_track.min(),native_track.max()),(-1,1))
        
        else :
            track,_ = load(track_path,sr=sampling_rate,mono=True,offset=t0,duration=duration)
            native_track, native_sr = load(track_path,sr=None,mono=True,offset=t0,duration=duration)
        
        #apply padding to have track duration every time --> TODO : change generation code or smthing else so that we can give any track
        r = len(track)%int(track_duration*sampling_rate)
        if r>0: #track duration less than track_duration
            pad = int(track_duration*sampling_rate)-r
            track = np.concatenate([track,np.zeros(pad)])
            
            r_nat = len(native_track)%int(track_duration*native_sr)
            pad_native = int(track_duration*native_sr)-r_nat
            native_track = np.concatenate([native_track,np.zeros(pad_native)])
        
        #zero-mean
        track = track - np.mean(track)
        native_track = native_track - np.mean(native_track)
        
        self.track = track 
        self.native_track = native_track
        self.native_sr = native_sr
        
        #chunk track
        self.track_chunks = self.segment_track(self.track,track_duration,self.sampling_rate, strategy=pre_segemntation,all=True)
        self.native_track_chunks = self.segment_track(self.native_track, track_duration, self.native_sr, strategy=pre_segemntation,all=True)
        
    
    def __len__(self):
        return len(self.track_chunks)    
    
    #unify 2 methods with get_chunk(self, index, native=True/False) -> getitem is get_chunks(index,native=False)
    def __getitem__(self, index):
        #get track
        track = self.track_chunks[index]
        duration = len(track)/self.sampling_rate
        
        #pad with zeros if track is not track_duration --> raises error in batching
        if duration<self.track_duration:
            pad = int((self.track_duration-duration)*self.sampling_rate)
            track = np.concatenate([track,np.zeros(pad)])
        
        #segment track into chunks
        chunks = self.segment_track(track,self.max_duration, self.sampling_rate,self.segmentation_strategy)
        
        slices = range(len(chunks))
        
        label = -1 #not used but needed for compatibility
        
        return chunks, label#, slices  
    
    def get_native_chunks(self,index):
        #get track
        track = self.native_track_chunks[index]
        #duration = len(track)/self.native_sr
        
        #pad with zeros if track is not track_duration --> raises error in batching
        #shouldnt be a problem since we pad at init
        # if duration<self.track_duration:
        #     pad = int((self.track_duration-duration)*self.native_sr)
        #     track = np.concatenate([track,np.zeros(pad)])
        
        #segment track into chunks
        chunks = self.segment_track(track,self.max_duration, self.native_sr, self.segmentation_strategy)
        
        return chunks
    
    
    def segment_track(self, track, max_duration, sampling_rate, strategy="uniform",all=False):
        
        track_duration=len(track) #in samples
        max_samples = int(max_duration*sampling_rate) #samples
        if strategy == "uniform":
            #segment input track into chunks of max_duration seconds
            N = int(track_duration//max_samples) #number of max_duration chunks
            if N > 0:
                chunks = [track[i*max_samples:(i+1)*max_samples] for i in range(N)]
                """ if all:
                    chunks.append(track[N*max_samples:]) """
                    
            else :
                chunks = [track]
        
        elif strategy=='sliding':
            hop_size_samples = int(self.hop_size*sampling_rate) #int(max_samples // 3)  # hop size is the third of the chunk size

            chunks = []
            for start in range(0, track_duration, hop_size_samples):
                end = start + max_samples
                
                if end > track_duration : #dont take last chunk if greater 
                    break
                
                chunk = track[start:end]
                
                chunks.append(chunk)
            
        elif strategy=='one':
            #pick one chunk from whole track
            t0 = np.random.uniform(0,track_duration-max_samples)
            chunks = [track[t0: t0+max_samples]]
        
        elif strategy == 'onset':
            #find onset points with backtrack for better segmentation
            y = resample(track,orig_sr=self.sampling_rate,target_sr=44100) #resample for essentia onset segmentation (optimized for 44.1khz)
            onset = detect_onsets(y,44100,with_backtrack=True) #in seconds
            onset = (onset*self.sampling_rate).astype(int) #in samples of original tracks
            
            #remove doubles
            onset = remove_duplicates(onset)
            #add 0 and duration for easier chunking
            onset = np.concatenate([[0],onset,[track_duration]])
            #process onsets
            onset = process_onsets(onset,None,max_samples) 
            #chunks (in seconds) cropped to max_duration
            chunks = [track[int(t0):int(t1)] for t0,t1 in zip(onset[:-1],onset[1:])]
        
        elif strategy=='None':
            chunks = [track]
            
        else :
            raise ValueError(f"Invalid segmentation strategy argument {strategy}.")
        
        return chunks
        
#cette class est une extension de de la v1 --> recoit pas une paire de root mais une liste de N roots --> generere N paires de stem vs mix
class MusicCouplingDatasetv2(Dataset):
    def __init__(self,roots, max_track_duration, max_chunk_duration, sampling_rate, 
                 segmentation_startegy="uniform", pre_segmentation='uniform', ignore_instrument=[], direction = "stem", verbose=False):
        # TODO generer doc
        
        super().__init__()
        
        self.sampling_rate=sampling_rate
        self.max_chunk_duration=max_chunk_duration
        self.max_track_duration=max_track_duration
        self.verbose=verbose
        self.segmentation_strategy = segmentation_startegy
        if ignore_instrument !=[] : raise ValueError("Ignore instrument done in pre-processing") #pas utilise ici : fait en amont
        self.ignore_instrument = ignore_instrument
        if direction == "bi" : raise NotImplementedError("Pas implemente dataset bidirectionnel...")
        self.direction = direction #direction couplage stem -> mix, mix -> stem, stem <-> mix
        
        if (self.max_chunk_duration*self.sampling_rate)%1!=0 :
            raise ValueError(f"max_duration = {self.max_chunk_duration} is incompatible with sampling rate.\n This error is due to a resolution problem.")
        
        #instanciate N MusicContainers
        self.containers = np.empty(len(roots),dtype=MusicContainerPostChunk)
        self.indexes = {} #dict containing container index and corresponding indexes
        for i,root in enumerate(roots):
            container = MusicContainerPostChunk(root,max_track_duration,max_chunk_duration,sampling_rate,
                                                segmentation_startegy,pre_segmentation=pre_segmentation,
                                                ignore_instrument=ignore_instrument)
            self.containers[i] = container
            start = self.indexes[i-1][-1]+1 if i>0 else 0 #length of last containers cumulated
            self.indexes[i] = [start,start+len(container)-1] #idx correspoinding to container i start at the end of last container and end at start + len -1 
        
        self.sort_containers() #CAREFULL ONLY WORKS IF ALL CONTAINERS HAVE TRACK NAMES WITH SIMILAR STRUCTURE. ok for clement cannone duos and trios, for moises should receive multi-track one by one
        
        #self.augment_inverse_tracks()
        
        l=[]
        for container in self.containers:
            l.append(len(container.audio_paths))
        assert len(set(l))==1, "There should be the same number of tracks for input and target."
        
    def __len__(self):
        coef = 1 if self.direction != "bi" else 2 #il y a 2x plus de couples si on fait couplage bidirectionnel
        return coef * min([len(container) for container in self.containers]) #take min value because some tracks dont have the exact same duration
    
    def __getitem__(self, index) :
        container_idx, chunk_idx = self._find_interval(index)
        other_idx = [idx for idx in self.indexes.keys() if idx!=container_idx]
        
        #randomly select a subset of the other idx (cf Diff-A-Riff)
        if len(other_idx)>1:
            #size = np.random.randint(1,len(other_idx))
            other_idx = np.random.choice(other_idx,size=(1,),replace=False)
        
        stem_chunks = self.containers[container_idx][chunk_idx][0] #gets the chunks of the stem : (N,samples)
        
        if self.segmentation_strategy!='onset':
            mix_chunks = np.mean([[chunks for chunks in self.containers[i][chunk_idx][0]] for i in other_idx],axis=0) #combine chunks : (N,samples)
            #mix_chunks = [np.interp(chunk,(chunk.min(),chunk.max()),(-1,1)) for chunk in mix_chunks]
        

        else :
            #if onset segmentation need to redo segmentation on whole mix... : find way to do this only once !!!!
            #not best solution but could do the getitem method, concat each track chunks, sum and the resegment... 
            # otherwise needs to modify postchunk to return unchunked tracks and chunk them here and has no sense if class used elsewhere...
            paths, starts, ends = zip(*[self.containers[i].track_chunks[chunk_idx][:3] for i in other_idx])
            start, end = starts[0], ends[0]
            duration = end-start
            mix = []
            for path in paths :
                track,_ = load(path,sr=self.sampling_rate,offset=start,duration=duration, mono=True)
                if duration < self.max_track_duration:
                    pad = int((self.max_track_duration-duration)*self.sampling_rate)
                    track = np.concatenate([track,np.zeros(pad)])
                mix.append(track)
            
            mix = np.mean(mix,axis=0)
            #mix = np.interp(mix,(mix.min(),mix.max()),(-1,1))
            mix_chunks = self.containers[0].segment_track(mix, 'onset')
                
        
        if self.direction == "stem":
            input_chunks = mix_chunks
            target_chunks = stem_chunks
        
        elif self.direction == "mix":
            input_chunks = stem_chunks
            target_chunks = mix_chunks 
        
        return input_chunks, target_chunks
      
    #CAREFUL THIS METHOD ONLY WORKS FOR PAIRING TRACKS OF SAME NAME !!!!
    #when going further in implementation find algorithm for correctly sorting conatiners by pair of tracks and then do mix-match augmentastion
    def sort_containers(self):
        for i in range(len(self.containers)):
            self.containers[i].track_chunks = sorted(self.containers[i].track_chunks, key = lambda x : (os.path.basename(x[0]),x[1]))   #sort by name then by start time

    
    
    def _find_interval(self,index):
        #metjod takes the index from __getitem__ and returns the corresponding key (container index) and new index in the container interval
        if index < 0 : index = len(self)+index #+ because index is negative
        for c_idx, (lower, higher) in self.indexes.items():
            if lower <= index <= higher :
                #index in c_idx interval
                new_idx = index-lower
                return c_idx, new_idx
        
        return ValueError(f"index {index} not in interval")            

class FineTuningDataset(Dataset):
    def __init__(self,guide_path, target_path, max_track_duration, max_chunk_duration, sampling_rate, 
                 segmentation_startegy="uniform", pre_segmentation='sliding', verbose=False):
    
        super().__init__()
        
        self.input_container = MusicContainerPostChunk(guide_path,max_track_duration,max_chunk_duration,sampling_rate,
                                                segmentation_startegy,pre_segmentation=pre_segmentation,verbose=verbose)
        
        self.target_container = MusicContainerPostChunk(target_path,max_track_duration,max_chunk_duration,sampling_rate,
                                                segmentation_startegy,pre_segmentation=pre_segmentation,verbose=verbose)
        
    def __len__(self):
        return min([len(self.input_container), len(self.target_container)])
    
    def __getitem__(self, index) :
        
        input_chunks = self.input_container[index][0] #gets the chunks of the stem : (N,samples)
        
        target_chunks = self.target_container[index][0]
                
        return input_chunks, target_chunks
    

class MusicCouplingContainer(Dataset):
    def __init__(self,roots : List[Tuple[Union[str,Path],Union[str,Path]]],
                 max_track_duration:float, max_chunk_duration:float, sampling_rate:int, 
                 segmentation_startegy:str="uniform", pre_segmentation='uniform',
                 ignore_instrument : List[str] = [], direction : str = "stem", verbose:bool=False):
        
        """_summary_

        Coupling Dataset Container : contains N MusicCouplingDataset for each pair of folder, tracks etc.
        Is used to handle different coupling datasets.
        
        Args:
            roots (List[Tuple[Union[str,Path],Union[str,Path]]]): roots contains the pairs of fodlers corresponding to coupled tracks. can also be a list of files
            max_track_duration (float): duration of uniform segmentation of tracks
            max_chunk_duration (float): max duration of a chunk for uniform or sliding window
            sampling_rate (int): _description_
            segmentation_startegy (str, optional): chunks segmentation strategy. Defaults to "uniform".
            ignore_instrument (List, optional) : list of instruments to ignore
            direction (str, optional) : how the coupling is learned. 'stem' is for mix->stem, 'mix' is for stem->mix and 'bi' is stemn<->mix
            verbose (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        
        #for each pair of tracks/folders create a MusicCouplingDataset
        self.containers=np.empty(len(roots),dtype=MusicCouplingDatasetv2)
        self.indexes = {} #dict containing container index and corresponding indexes
        for i,couple_roots in enumerate(roots):
            container = MusicCouplingDatasetv2(couple_roots,max_track_duration,max_chunk_duration,sampling_rate,
                                               segmentation_startegy,pre_segmentation,
                                               ignore_instrument,direction,verbose)
            self.containers[i]=container
            start = self.indexes[i-1][-1]+1 if i>0 else 0 #length of last containers cumulated
            self.indexes[i] = [start,start+len(container)-1] #idx correspoinding to conmtainer i start at the end of last container and end at start + len -1 
    
    def __len__(self):
        return sum([len(container) for container in self.containers]) #get the max index in the list

    def __getitem__(self,idx):
        c_idx, new_idx = self._find_interval(idx) 
        #get pair of tracks from corresponding container
        input_chunks, target_chunks = self.containers[c_idx][new_idx]
        
        return input_chunks, target_chunks   
    
    
    def _find_interval(self,index):
        #metjod takes the index from __getitem__ and returns the corresponding key (container index) and new index in the container interval
        if index < 0 : index = len(self)+index #+ because index is negative
        for c_idx, (lower, higher) in self.indexes.items():
            if lower <= index <= higher :
                #index in c_idx interval
                new_idx = index-lower
                return c_idx, new_idx
        
        return ValueError(f"index {index} not in interval")    
        



class Fetcher:
    def __init__(self, loader, device=None):
        self.loader=loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device==None else device
    
    def _fetch_inputs(self)-> Union[torch.Tensor, dict]:
        #method to fetch next set of inputs
        try:
            #try to fectch next inputs
            inputs = next(self.iter_loader)
        except (AttributeError, StopIteration):
            #if self.iter_loader not already instantiated or end of loader
            self.iter_loader = iter(self.loader)
            inputs = next(self.iter_loader)
        
        return inputs
    
    def __len__(self):
        return len(self.loader)
    
    def __next__(self) -> Munch:
        inputs = self._fetch_inputs()
        
        #pass inputs to cuda and as a Munch
        inputs_device = Munch({key : item.to(self.device) if type(item)==torch.Tensor else [elem.to(self.device) for elem in item] for key,item in inputs.items()}) #handle list inputs in case onset segmentation is encountered
        
        return inputs_device

 

@dataclass 
class MusicDataCollator:
    
    unifrom_chunks : bool = True #used to handle datacollation with inhomogenous shapes
    sampling_rate : int = 16000 #in case from_onset is true
    with_slices: bool = False
    mask_prob : float = 0.0 #probability for a timestep to be masked
    mask_len : int = 0 #span for a single mask
    
    def _normalize_chunks(self, batched_chunks : tuple):
        #method to normalize with Z-score (0 mean and unit variance) each chunk from the batched set of chunks
        #there are N chunks per element of the list and we normalize each of those chunks
        
        #normalized_chunks = tuple((chunks-np.mean(chunks,axis=1,keepdims=True))/(np.std(chunks,axis=1,keepdims=True)+1e-5) for chunks in batched_chunks)
        
        normalized_chunks = []
        
        for chunks in batched_chunks:
            chunks_cat = np.concatenate(chunks) #needed for inhomogenous chunks (i.e. onset segmentation)
            mean = np.mean(chunks_cat)
            std = np.std(chunks_cat)
            normalized_chunks.append([(chunk-mean)/(std+1e-9) for chunk in chunks])
        
        #normalized_chunks = tuple(normalized_chunks)
        
        return normalized_chunks   
    
    #TODO : THIS METHOD CAN BE UNIFIED FOR UNIFORM AND INHOMOGENOUS CHUNKS !!!
    def _pad_chunks(self, batched_chunks : tuple):
        #this method applies a padding to the 1st dimension of the chunks with respect of the max number of chunks in the batch
        #new : also applies apdding to samples dimension and saves the mask
        
        #find biggest N (chunks)
        N_max = len(max(batched_chunks, key = lambda x : len(x)))
        B=len(batched_chunks)
        
        if self.unifrom_chunks:
            
            #samples dim paddign mask is all false when uniform chunking (obvious, could also give None)
            samples_padding_mask = np.zeros_like(batched_chunks) #(B, max_chunks, max_samples)
            
            batched_chunks = np.array(batched_chunks)
            L = batched_chunks[0].shape[1] #assuming all chunks have the same number of samples
            #pad the other set of chunks to N_max AND KEEP THE PADDING MASK
            chunks_padding_mask=np.zeros((B,N_max)) #0 for attending and 1 for not attending (aka where it is padded). shape = (batch, max_chunks)
            
            batched_padded_chunks=np.zeros((B,N_max,L)) #batch size, max_chunks, max_samples
            
            for i,chunks in enumerate(batched_chunks):
                pad_len = N_max-chunks.shape[0] #number of times to append max_samples chunks
                if pad_len==0 : 
                    #no padding
                    batched_padded_chunks[i]=chunks
                    continue
                pad = np.zeros((pad_len,L)) #number of chunks with max_samples
                padded_chunks = np.concatenate([chunks,pad])
                chunks_padding_mask[i,-pad_len:]=1
                batched_padded_chunks[i]=padded_chunks
                        
        else :
            
            #pad sample dim to max_samples and save padding mask
            #to enable tensor input
            
            L_max = max(len(x) for chunks in batched_chunks for x in chunks) #max number of samples (longest chunk in whole batch)
            
            L_max = max(L_max, int(MIN_RESOLUTION*self.sampling_rate)) #minimum num of samples to pass the backbone
            
            #pad the other set of chunks to N_max AND KEEP THE PADDING MASK
            chunks_padding_mask=np.zeros((B,N_max)) #0 for attending and 1 for not attending (aka where it is padded). shape = (batch, max_chunks)
            #pad chunks to L_max in samples dimension
            samples_padding_mask = np.zeros((B,N_max,L_max))
            
            batched_padded_chunks=np.zeros((B,N_max,L_max)) #init as empty list because data has inhomogenous shape
            
            for i,chunks in enumerate(batched_chunks):
                #pad chunk dim
                c_pad_len = N_max - len(chunks) 
                if c_pad_len != 0:
                    c_pad = np.zeros((c_pad_len,L_max)) #num chunks of L_max samples to pad
                    chunks.extend(c_pad) #extend list of chunks with chunk padding
                    chunks_padding_mask[i,-c_pad_len:]=1 #update chunk padding maks
                                 
                for j,chunk in enumerate(chunks) :
                    #pad sample dim
                    pad_len = L_max - len(chunk)
                    if pad_len == 0 : 
                        batched_padded_chunks[i][j] = chunk 
                        continue 
                    pad = np.zeros(pad_len)
                    padded_chunk = np.concatenate([chunk,pad])
                    samples_padding_mask[i,j,-pad_len:]=1
                    batched_padded_chunks[i,j] = padded_chunk            
        
        return batched_padded_chunks, samples_padding_mask, chunks_padding_mask 
    
    def _mask_input_chunks(self, batched_chunks, padding_mask):
        B, seq_len = batched_chunks.shape[:-1] #we are interested in the shape after collapse and quantize --> seq_len = chunks
        
        if self.mask_len<1 :
            raise ValueError("if 'mask_prob'>0. then mask len has to be greater than 0 too.")
        
        # if self.mask_prob*seq_len / self.mask_len < 1 :
        #     raise ValueError("Combination of 'mask_prob' and 'mask_len' is too big for the input sequence")
            
        
        mask_indices = compute_mask_indices((B,seq_len),torch.from_numpy(padding_mask),mask_prob=self.mask_prob,mask_length=self.mask_len,min_masks=1)
        
        return mask_indices
        
        
    def process_chunks(self, batched_chunks : tuple):
        normalized_chunks = self._normalize_chunks(batched_chunks) #0 mean, 1 variance
        padded_chunks, samples_padding_mask, chunks_padding_mask = self._pad_chunks(normalized_chunks) #pad to max sequence length
        
        return padded_chunks, samples_padding_mask, chunks_padding_mask
    
    
    def __call__(self,batch) -> Munch[torch.Tensor,torch.Tensor,Tuple[torch.Tensor,torch.Tensor],torch.Tensor]:
        
        slices=[] # for compatibility
        if self.with_slices:
            input_chunks,labels,slices = zip(*batch)
        else : 
            input_chunks,labels = zip(*batch)
        
        #process chunks
        input_chunks, samples_padding_mask, chunks_padding_mask=self.process_chunks(input_chunks)
        
        mask_indices = np.zeros(input_chunks.shape[:-1]) #init to all false
        if self.mask_prob>0:
            mask_indices = self._mask_input_chunks(input_chunks,chunks_padding_mask)
        

        #convert to torch tensors (valid for uniform or not because padded for batching anyways)
        input_chunks = torch.tensor(input_chunks,dtype=torch.float)
        
        
        samples_padding_mask = torch.tensor(samples_padding_mask, dtype=torch.bool)
        chunks_padding_mask = torch.tensor(chunks_padding_mask, dtype=torch.bool)
        labels = torch.tensor(labels,dtype=torch.float)
        mask_indices = torch.tensor(mask_indices,dtype=torch.bool)
        slices = torch.tensor(slices,dtype=torch.int)
        
        
        return Munch(src=input_chunks,
                     src_padding_masks = [samples_padding_mask,chunks_padding_mask],
                     mask_indices = mask_indices,
                     labels = labels,
                     slices=slices)
                         
    

    
#OPTIMIZE THIS STRUCTURE TO RUN FASTER FOR NOW IT IS QUITE SLOW
# TODO : handle T_context here rather than in the model. T_context is in seconds and max_len is in frames (equal to max chunks). crop end of track to have chunks<=max_len
@dataclass
class DataCollatorForCoupling(MusicDataCollator):
    
    def __call__(self, batch) -> Munch[torch.Tensor,torch.Tensor,Tuple[torch.Tensor,torch.Tensor],Tuple[torch.Tensor,torch.Tensor],torch.Tensor]:
        input_chunks, target_chunks = zip(*batch)
        
        #input and target chunks are lists of len = batch size and eahc element is of shape N,max_samples but N may vary
        #the goal is to pad the 1st dimension to have same number of chunks per batch
        #to do so we padd with max_samples * N_max-N on each element and normalize it
        #we have to extract the the corresponding padding mask in order to not compute the loss over padded sequences
        input_chunks, input_s_padding_mask, input_c_padding_mask=self.process_chunks(input_chunks)
        target_chunks, target_s_padding_mask, target_c_padding_mask=self.process_chunks(target_chunks)
        
        
        #compute mask time indices
        mask_indices = np.zeros(input_chunks.shape[:-1]) #init to all false
        if self.mask_prob>0:
            mask_indices = self._mask_input_chunks(input_chunks,input_c_padding_mask)
            
        
        #convert to torch tensors (valid anyways with new padding)
        input_chunks = torch.tensor(input_chunks,dtype=torch.float)
        target_chunks = torch.tensor(target_chunks, dtype=torch.float)
        
        
        input_s_padding_mask = torch.tensor(input_s_padding_mask, dtype=torch.bool)
        input_c_padding_mask = torch.tensor(input_c_padding_mask, dtype=torch.bool)
        
        target_s_padding_mask = torch.tensor(target_s_padding_mask, dtype=torch.bool)
        target_c_padding_mask = torch.tensor(target_c_padding_mask, dtype=torch.bool)
        
        #src_mask = torch.tensor(src_mask, dtype=torch.bool)
        #tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool)
        
        mask_indices = torch.tensor(mask_indices,dtype=torch.bool)
        
        return Munch(src = input_chunks, 
                     tgt = target_chunks,
                     #src_mask=src_mask,
                     #tgt_mask=tgt_mask,
                     src_padding_masks = [input_s_padding_mask,input_c_padding_mask],
                     tgt_padding_masks = [target_s_padding_mask,target_c_padding_mask],
                     src_mask_indices = mask_indices)
        

#%%
# import os

# path = "../data/moisesdb"#"/moisesdb_v0.1"
# path_test = "../data/Examples"

# # path=path_test

# # folders = os.listdir(path)
# # audio_files=[]
# # for folder in folders:
# #     sub_path=os.path.join(path,folder)
# #     if os.path.isdir(sub_path):
# #         sub_folders=os.listdir(sub_path)
# #         for sub_folder in sub_folders:
# #             # print(sub_folder)
# #             # print(listdir(os.path.join(sub_path,sub_folder)))
# #             audio_files+=listdir(os.path.join(sub_path,sub_folder))
# #     else :
# #         audio_files+=[sub_path]

# audio_files=[]
# for root, directories, files in os.walk(path):
#     for file in files:
#         if file.lower().endswith(('.wav', '.aif')):  # Consider lowercase extensions
#             # print(root.split("\\")[-1],"\n", file)    
#             file_path = os.path.join(root, file)
#             label = root.split("\\")[-1] 
#             if label not in moisesdb_labels:
#                 label = "UNK" #unknown label (aka instrument)
#             #audio path + instrument 
#             audio_files.append([file_path, label])
    
    