#%%
from utils.utils import lock_gpu
device=lock_gpu()[0][0]
from MusicDataset.MusicDataset_v2 import MusicContainer, INSTRUMENT_LABELS
from wav2vec2.wav2vec2_utils import DataCollatorForWav2Vec2, Fetcher
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from architecture.Encoder import Backbone
from architecture.Model import build_backbone,build_quantizer
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from fairseq import checkpoint_utils
from tqdm import tqdm
from sklearn.manifold import TSNE #dim reduction
from sklearn.decomposition import PCA
from umap import UMAP
from utils.utils import prGreen,prRed,prYellow
import soundfile as sf
import os
from typing import Union
from pathlib import Path
import math
#%%
def generate_data(num_samples, samples_dir, save_npy):
# Lock GPU

    #device=lock_gpu()[0][0]
        
        
    # load models and feature_extractor
    
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
   
    baseline_model = build_backbone("../w2v_music_checkpoint.pt","w2v",mean=False,pooling=False)
    baseline_model.to(device)
    
    #w2v backbone adapted for classif
    adapted_checkpoint="runs/classif_adapt/classif_adapt.pt"
    state_dict=torch.load(adapted_checkpoint,map_location=torch.device("cpu"))
    model =  build_backbone("../w2v_music_checkpoint.pt","w2v",mean=False,pooling=False)
    adapted_model = Backbone(model.backbone,"w2v",mean=True) #adapted model used mean of latents accross time
    
    #list of state dicts if saved whole classifier
    if isinstance(state_dict,list):
        #si .pt contient model
        adapted_model.load_state_dict(state_dict[0])
        
    
    else: #older trained models only had backbone in state dict
        #si .pt contient state_dict -> instancier model avant de load
        adapted_model.load_state_dict(state_dict)
    
    adapted_model.to(device)
    
     #w2v speech
    speech_model=Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    speech_model.to(device)
   
    # Instanciate dataset, datacollator and dataloader
    #dataset arguments
    root = "/data3/anasynth_nonbp/bujard/data/moisesdb_v2/train"
    max_duration=5.0
    sampling_rate=feature_extractor.sampling_rate
    segmentation_strategy="one"

    #dataset containing all the chunks
    eval_ds = MusicContainer(root, max_duration, sampling_rate, segmentation_strategy, non_empty=True) #remove most of the empty chunks

    #DataCollator
    collate_fn = DataCollatorForWav2Vec2(baseline_model.backbone, feature_extractor, split="test") #same datacollator

    #dataloader
    batch_size=8
    eval_loader=DataLoader(eval_ds,batch_size,collate_fn=collate_fn, shuffle=True)

    eval_fetcher = Fetcher(eval_loader)
    eval_fetcher.device = device #move fetcher to corresponding device
    #
    data_baseline={"latents":[], "labels":[]}
    data_adapted={"latents":[], "labels":[]}
    data_speech={"latents":[], "labels":[]}
    
    baseline_model.eval()    
    adapted_model.eval()
    speech_model.eval()
    
    #model.train()
    #progress_bar = tqdm(range(len(eval_loader)))
    idx=0
    #num_samples=1000
    with torch.no_grad():
        for i in tqdm(range(num_samples//batch_size)):
            inputs=next(eval_fetcher)
            
            #speech
            outputs = speech_model(inputs.x, output_hidden_states=True)
            z_speech = outputs.hidden_states[-1]
            z_speech=z_speech[:,-1,:] #keep one time step for all batches
           

            z_base = baseline_model(inputs.x)[:,-1,:]
            
            
            #adapted
            z_adapt=adapted_model(inputs.x)
            z_adapt=z_adapt.squeeze(1) #remove time dimension equal to 1
           
            
            instruments = inputs.instruments
            
            
            data_baseline["labels"].append(instruments.cpu().detach().numpy())
            data_baseline["latents"].append(z_base.cpu().detach().numpy())
            
            
            data_adapted["labels"].append(instruments.cpu().detach().numpy())
            data_adapted["latents"].append(z_adapt.cpu().detach().numpy())
           
            
            data_speech["labels"].append(instruments.cpu().detach().numpy())
            data_speech["latents"].append(z_speech.cpu().detach().numpy())
           
            
            #save chunks as audio files
            chunks=inputs.x.cpu()
            if save_npy:
                for chunk in chunks:
                    fname=f"{samples_dir}/chunk_{idx}.wav"
                    sf.write(fname,chunk,sampling_rate)
                    idx+=1
                    #data["chunks"].append(fname)
            
            #progress_bar.update(1)
            

    #save data to npy
    if save_npy:
        np.save("CataRT_data/dim_reduction/data_baseline.npy",data_baseline, allow_pickle=True)
        np.save("CataRT_data/dim_reduction/data_adapted.npy",data_adapted, allow_pickle=True)
        np.save("CataRT_data/dim_reduction/data_speech.npy", data_speech, allow_pickle=True)
    
    return data_baseline, data_adapted, data_speech
#%%    
def dim_reduction(data, reduction, config, save_npy, return_red=False):
    assert reduction in ["t-sne","pca","umap"], f"reduction argument is expected to be one of [t-sne,pca,umap], not {reduction}"
    
    #get latent vectors and instruments
    X=np.array(data['latents']) #as np array for reduction algorithms
    Y=np.array(data['labels'])
    
    #if still batch dimension
    if len(X.shape)==3:
        X=np.concatenate(X,axis=0) #could use reshape but less control over "what happens" during the reshape than concatenate
        Y=np.concatenate(Y,axis=0)
    
    prYellow(f"Dimensionality reduction using {reduction} algorithm")

    if reduction=="t-sne":
        red = TSNE(n_components=2, perplexity=50.0,learning_rate='auto',
                    init='pca', random_state=42) 
        
        X_embedded = red.fit_transform(X)

    elif reduction=="pca":
        red =PCA(n_components=2)
        X_embedded=red.fit_transform(X)

    elif reduction=="umap":
        red = UMAP()
        X_embedded = red.fit_transform(X)
    
    reduced_data={'latents':X_embedded,'labels':Y} #keep labels in case. chunks are all the same and are stored in CataRT data/samples
        
    if save_npy:
        prGreen("Saving reduced data...")
        np.save(f"reduced_data_{config}_{reduction}.npy", reduced_data, allow_pickle=True)
    
    if return_red :
        return reduced_data, red
    
    return reduced_data
#%%
def save_as_txt(data : dict, samples_dir : Union[str,Path], fname : str, task : str):
    assert fname.endswith(".txt"), f"fname has to end with .txt extention"
    assert task in ["dim_reduction", "quantized"]
    
    if task == "dim_reduction":
        #convert to np array if necessary (depending from where data was generated) and combine batch dimension
        if type(data['latents'])==list or len(data['latents'].shape)==3:
            X=np.concatenate(data['latents'],axis=0)
            Y=np.concatenate(data['labels'],axis=0)
            data = {key:item for key,item in zip(data.keys(),[X,Y])}
        
        
        #take all samples from directory and be sure they are ordered by idx
        #latents and labels are stored in that order so we need to be sure the correct data is extracted from the correspondig file
        chunks=sorted(os.listdir(samples_dir),key=lambda x: int(x[6:-4])) 

        #write a file with space separated values
        head = ["FileName"]+ [f"d{i}" for i in range(len(data['latents'][0]))] + ["instrument"]
        
        lines = []
        for chunk,latent,instrument in zip(chunks,data['latents'],data['labels']):
            chunk=chunk.split("/")[-1] #not necessary but good precaution
            line=[chunk]+ [str(d) for d in latent] + [str(instrument)]
            lines.append(line)

    elif task == "quantized":
        chunks = sorted(os.listdir(samples_dir))
        #write a file with space separated values
        head = ["FileName"]+ ["quantized_idx"] + ["instrument"]
        
        lines = []
        for chunk,idx,instrument in zip(chunks,data['idx'],data['instrument']):
            #chunk=os.path.join(task,chunk) 
            line=[chunk] + [str(idx)] + [str(instrument)]
            lines.append(line)   
        
    #text = head+values
    with open(fname,'w') as f:
        for col in head:
            f.write(col+" ") #space separated value
        f.write("\n") #newline
        for line in lines:
            for col in line:
                f.write(col+" ") #space separated value
            f.write("\n")
#%%
def viz(X_embedded,Y,name, reduction_type):
    X_group = [X_embedded[np.where(Y==i)] for i in set(Y)]

    plt.figure(figsize=(10,10), dpi=150)
    for group, class_id in zip(X_group,set(Y)):
        label=INSTRUMENT_LABELS[class_id]
        if label == "wind" : continue
        plt.scatter(group[:,0], group[:,1], label=label)
    
    plt.legend(loc="upper left")
    plt.xlabel(r"$z_{1}$", weight='bold')
    plt.ylabel(r"$z_{2}$", weight='bold')
    plt.title(f"Latent space visualisation with {reduction_type}")
    plt.savefig(f"Latent_space_{name}.png") #remove data_ and npy
    plt.show()    
#%%
def reduce_and_viz(data_file):
    # load samples
    data = np.load(data_file, allow_pickle=True).item()
    
    #reshape array as NxD
    data['latents'] = np.concatenate([arr for arr in data['latents']],axis=0)
    data['labels'] = np.concatenate([arr for arr in data['labels']])
    
    num_examples=min(1000,len(data['labels']))
    np.random.seed(42) #reproductibility
    idxs = np.random.choice(range(len(data['labels'])),
                            size=num_examples,
                            replace=False)
    X = data['latents'][idxs]
    Y = data['labels'][idxs]
    # dimensionality reduction with t-SNE

    #high perplexity cause high dimension data, init pca to help convergence
    #random_state for reproductibility
    tsne = TSNE(n_components=2, perplexity=50.0,learning_rate='auto',
                init='pca', random_state=42, verbose=1) 

    X_embedded = tsne.fit_transform(X)


    # Visualize projected data points 
    #group data points by instrument
    name=data_file.split("/")[-1][:-4]
    viz(X_embedded,Y,name)


def generate_quantized_data(roots,
                            max_duration,
                            vocab_size,
                            num_samples,
                            ignore,
                            segment='one',
                            non_empty = True,
                            save_data=True,
                            device=None):
    #not lock multiple gpus for every run
    if device == None : device=lock_gpu()[0]
    
    #build modules
    backbone = build_backbone("/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt","w2v",mean=False,pooling=False,fw="fairseq")
    backbone.to(device)
    
    quantizer = build_quantizer(backbone.dim,vocab_size,learnable_codebook=False)
    quantizer.to(device)
    
    #build dataset
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    sr=feature_extractor.sampling_rate
    #dataset containing all the chunks
    ds = MusicContainer(roots, max_duration, sr, segment, non_empty=non_empty, ignore_instrument=ignore)

    #DataCollator
    collate_fn = DataCollatorForWav2Vec2(backbone.backbone, feature_extractor, split="test") #same datacollator

    #dataloader
    batch_size=8
    loader=DataLoader(ds,batch_size,collate_fn=collate_fn, shuffle=True)

    fetcher = Fetcher(loader)
    fetcher.device = device #move fetcher to corresponding device
    
    data = {"original":[],"mean":[], "centroids":quantizer.centers.numpy(force=True),"quantized":[],"idx":[], "instrument":[]}
    
    chunks_data=[]
    if save_data:
        sample_dir=f"CataRT_data/samples/quantized/vocab_{vocab_size}/res_{max_duration}s"
        data_dir = f"CataRT_data/quantized/vocab_{vocab_size}/res_{max_duration}s"
        os.makedirs(sample_dir,exist_ok=True) #overwrite existent dir if necessary
        os.makedirs(data_dir,exist_ok=True)
    
    
    chunk_idx = 0
    if num_samples == -1 : num_samples = len(loader.dataset) #take all
    #else : num_samples = min(len(loader),math.ceil(num_samples/batch_size))
    
    backbone.eval()
    quantizer.eval()
    
    with torch.no_grad():
        for i in tqdm(range(num_samples//batch_size)):
            #get data
            inputs=next(fetcher)
            
            #pass through backbone
            z = backbone(inputs.x) #(B,L,D)
            
            
            #collapse 
            z_c = torch.mean(z,dim=1) #(B,D)
            
            #quantize
            z_q, idx, _ = quantizer(z_c)
            
            
            #save chunks as audio files
            chunks=inputs.x.cpu()
            if save_data:
                for chunk in chunks:
                    
                    fname=f"{sample_dir}/chunk_{chunk_idx:03}.wav"
                    sf.write(fname,chunk,sr)
                    chunk_idx+=1
            
            else : chunks_data.extend(chunks.cpu().detach().numpy())
            
            
            data["original"].extend(z.cpu().detach().numpy())
            data["mean"].extend(z_c.cpu().detach().numpy())
            data["quantized"].extend(z_q.cpu().detach().numpy())
            data["idx"].extend(idx.cpu().detach().numpy())
            data["instrument"].extend(inputs.instruments.cpu().detach().numpy())
    
    if save_data:
        #save data
        np.save(f"{data_dir}/data.npy",data,allow_pickle=True)
    
    else :
        return data, chunks_data


if __name__=="__main__":
    
    D_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A1"
    D_A2="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A2"
    T_A2 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A2"
    T_A3 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A3"
    moisesv2 = "/data3/anasynth_nonbp/bujard/data/moisesdb_v2/train"
    roots = [D_A1,D_A2,T_A2,T_A3,moisesv2]

    max_duration=0.5
    vocab_size = 32
    num_samples = 200

    generate_quantized_data(moisesv2,max_duration,vocab_size,num_samples)
    #for dim red viz
    """ 
    data_path = "CataRT_data/dim_reduction/baseline/reduced_data_baseline_t-sne.npy"
    data = np.load(data_path,allow_pickle=True).item()
    X,Y=data['latents'],data['labels']
    viz(X,Y,"baseline","t-SNE") 
    """
     
    
    #dont run again
    if False :
        #args
        num_samples=1000
        samples_dir="CataRT_data/samples" #carefull may overwrite previous data
        #save_npy=False
        
        #generate data
        data_base, data_adapt, data_speech = generate_data(num_samples, samples_dir, save_npy=True)
        #dim reduction
        reductions = ["t-sne","pca","umap"]
        for config,data in zip(["baseline","with_adaptation","speech"],[data_base,data_adapt,data_speech]):
            for reduction in reductions:
                data_red = dim_reduction(data,reduction,config,save_npy=True)
                #save as .txt for CataRT interface
                fname=f"reduced_data_{config}_{reduction}.txt"
                save_as_txt(data_red,samples_dir,fname)
    
    """
    task="viz"
    #data_file="data_adapted.npy" #"results/latent_viz/data_base.npy"
    data_file="CataRT data/with_adaptation/reduced_data_with_adaptation_umap.npy"
    
    if task=="generate":
        generate_data()
    
    elif task=="red_viz":
        reduce_and_viz(data_file)
    
    elif task=='viz':
        data = np.load(data_file, allow_pickle=True).item()
        X=data['latents']
        Y=data['labels']
        name=data_file.split("/")[-1][:-4]
        viz(X,Y,name)
    """
    
        
        
        
        