#%%
#from utils.utils import lock_gpu
#device = lock_gpu()[0][0]
from utils.utils import build_coupling_ds
from architecture.Model import load_model_checkpoint
from utils.coupling_ds_generator import extract_all_groups
import torch
from tqdm import tqdm
#%%
""" 
#build model
ckp = "/data3/anasynth_nonbp/bujard/DICY2/runs/coupling/0.5_512.pt"
model = load_model_checkpoint(ckp)
model.to(device)
model.eval()
#%%

#build ds
D_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/val/A1"
D_A2="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/val/A2"
T_A1 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A1"
T_A2 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A2"
T_A3 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A3"

moisesdb_val = "../data/moisesdb_v2/val"
moises_tracks = extract_all_groups(moisesdb_val)

val_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]+moises_tracks

try :
    track_duration = model.track_size
    chunk_duration = model.chunk_size
except AttributeError:
    track_duration = 30.
    chunk_duration = 0.5

val_fetcher = build_coupling_ds(val_roots,8,
                                    track_duration,chunk_duration,
                                    segmentation_strategy=model.segmentation,
                                    pre_segmentation="sliding",
                                    SAMPLING_RATE=16000,
                                    direction="stem",distributed=False)
val_fetcher.device = device
 """
#%%

#k=1 

def compute_similarity(tgt : torch.Tensor, tgt_idx : torch.Tensor, logits : torch.Tensor, codebook, eos_idx, k) -> torch.Tensor:
    #remove sos
    tgt_out = tgt[:,1:] # (B,L,D)
    tgt_idx = tgt_idx[:,1:] #(B,L)
    
    #remove everything at and after eos token --> only vocab tokens 
    eos_pos = torch.argwhere(tgt_idx==eos_idx) #(B,2)
    
    tgt_out_flat = torch.tensor([],device=tgt.device) # (L',D)
    logits_flat = torch.tensor([],device=logits.device) #(L',C)
    for pos in eos_pos:
        tgt_out_flat = torch.cat([tgt_out_flat,tgt_out[pos[0],:pos[1]]])
        logits_flat = torch.cat([logits_flat,logits[pos[0],:pos[1],:]])
    
    L,D = tgt_out_flat.size()
    #top-k search
    topK_idx = torch.topk(logits_flat[:,:-3],k,dim=-1)[1] #(L',k) and remove 3 last classes (sos,eos,pad)

    #compute similarity between tgt and pred vectors 
    sim=torch.tensor([0.,0.],device=logits.device)
    for gt,idxs in zip(tgt_out_flat,topK_idx):
        #idxs is (k,) and gt is (D,)
        preds = codebook[idxs] #(k,D)

        c = torch.cosine_similarity(gt.unsqueeze(0),preds)

        sim[0] += c.mean()
        sim[1] += c.std()
    
    sim = sim/L #mean 
    return sim

def compute_topK_similarity(model,eval_fetcher,k):
    
    device = model.device
    
    codebook = model.vocab_embedding_table
    eos_idx = model.special_tokens_idx["eos"].to(device)
    sos_idx = model.special_tokens_idx["sos"].to(device)
    pad_idx = model.special_tokens_idx["pad"].to(device)

    cosine_sim = torch.tensor([0.,0.],device=device) #mean, std
    #inference
    print(f"Computing similarity for k = {k}")
    with torch.no_grad():
        for i in tqdm(range(len(eval_fetcher))):
            inputs = next(eval_fetcher)
            src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices = inputs.values()
            
            #compute output
            logits, tgt, tgt_idx, _ = model(src, tgt, src_pad_mask, tgt_pad_mask)
            
            #compute similarity with topK predictions
            sim = compute_similarity(tgt,tgt_idx,logits,codebook,eos_idx,k)
                
                
            cosine_sim+=sim
                #print(sim)
                
    cosine_sim/=len(eval_fetcher)     
    #print(cosine_sim.numpy(force=True))   
    return cosine_sim.numpy(force=True)
        
    #%%
        
        
        