from architecture import SimpleSeq2SeqModel, build_backbone
import math
import numpy as np
from typing import List
from utils.utils import lock_gpu, prYellow, build_coupling_ds
from MusicDataset import MusicContainerPostChunk, MusicDataCollator, Fetcher
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import os
from trainer import Seq2SeqTrainer
import torch
from argparse import ArgumentParser


DEVICE = lock_gpu()[0][0]


def trainVQ(codebook_size : int, chunk_duration : float, tracks : List[str]):
    #fonction reçoit taille du codebook, le chunk size et la liste des pistes utilisées pour l'entrainement du moodèle
    
    #codebook_size=16

    dim=768
    output_final_proj = False

    #model
    prYellow("Loading model from checkpoint...")
    checkpoint="../w2v_music_checkpoint.pt"
    model = build_backbone(checkpoint,type="w2v",mean=False,pooling=False,output_final_proj=output_final_proj)
    model.to(DEVICE)
    model.freeze()
    model.eval()


    #dataset
    track_duration=15
    sr=16000
    segmentation_strategy="sliding" 
    ds = MusicContainerPostChunk(tracks, track_duration, chunk_duration, sr, segmentation_strategy)
    
    # dataloader
    batch_size=64
    collate_fn = MusicDataCollator()
    loader = DataLoader(ds,batch_size,shuffle=True,collate_fn=collate_fn)
    fetcher=Fetcher(loader, device=DEVICE)


    # kmeans
    #k_means_batch_size=int(batch_size*max_duration*sr/400) #corresponds to the number of samples per batch. 400 is approx the subsample coeficient (1 sample is 0.025s)
    k_means_batch_size=int(batch_size*(track_duration/chunk_duration)) #total samples per batch
    
    k_means=MiniBatchKMeans(codebook_size,batch_size=k_means_batch_size,random_state=42)

    #fit kmeans to data
    epochs=5
    bar=tqdm(range(epochs*len(fetcher)))
    old_centers = np.zeros((codebook_size,model.dim))
    new_dist = 0
    delta = 1000
    end=False
    for epoch in range(epochs):
        prYellow(f"Epoch {epoch+1}/{epochs}...")
        for iter in range(len(fetcher)):
            inputs = next(fetcher)
            x = inputs.src #get batched data (B,chunks,samples)
            masks = inputs.src_padding_masks
            
            #reshape as (B*chunks,samples)
            x = x.reshape(-1,x.size(-1))
            
            #pass through model
            z = model.forward(x,masks[0]) # (B*chunks,T,dim)
            
            #reshape as N,latent_dim and to numpy for sklearn compatibility
            z = z.reshape(-1,z.size(-1)).numpy(force=True) #(B,dim)
            
            #partial fit kmeans
            k_means.partial_fit(z)
            
            #compute distance between old an new centers  
            dist = np.linalg.norm(old_centers-k_means.cluster_centers_,axis=1)
            delta = abs(new_dist-dist.mean())
            new_dist = dist.mean()
            
            #print(new_dist, delta)

            old_centers=k_means.cluster_centers_.copy()
            
            bar.update(1)     
            
            #end after one epoch or more if centers have converged
            if epoch>0 and delta<1e-4:
                end=True
                break  
        
        #save centers for later use as VQ at end of epoch
        centers=k_means.cluster_centers_
        prYellow(f"Saving kmeans centers...")
        np.save(f"myVQ/kmeans_centers_{codebook_size}_{chunk_duration}s.npy",centers,allow_pickle=True)
        if end:
            break

def build_model(args):
    
    #BACKBONE
    pretrained_bb_checkpoint = "../w2v_music_checkpoint.pt"
    bb_type="w2v"
    dim=768 #args.dim
    pre_post_chunking = "post" #args.pre_post_chunking
    freeze_backbone= True #args.freeze_backbone 

    vocab_size = args.vocab_size

    #VQ
    #dim=768  #quantizer output dimension. if different than backbone dim must be learnable codebook (becomes an nn.Embedding layer to be learned)
    learnable_codebook= False #args.learnable_cb#args.learnable_cb #if the codebooks should get closer to the unquantized inputs
    restart_codebook= False #args.restart_codebook #update dead codevectors
    #if restart_codebook and not learnable_codebook: prRed("restart codebook without learnable codebook") 
    
    chunk_duration = args.chunk_duration #[sec] #equivalent to resolution for decision input
    track_duration = 15.0 #args.track_duration #(30/0.5)*MAX_CHUNK_DURATION #[sec] comme ca on a tojours des sequences de 60 tokens dans decision #[sec]
    segmentation= args.segmentation
    

    #POS ENCODING
    #div_term = MAX_CHUNK_DURATION if SEGMENTATION_STRATEGY in ['uniform','sliding'] else MIN_RESOLUTION
    max_len = int(math.ceil(track_duration/chunk_duration)+1 + 10) #max len is the max number of chunks + some overhead #max(200,int(math.ceil(MAX_TRACK_DURATION/div_term)+1 + 3)) #gives the max number of chunks per (sub-)track +1 cuz of padding of last track chunk +3 for special tokens
    
    encoder_head=args.encoder_head #COLLAPSE method
    condense_type=args.condense_type if encoder_head!='mean' else None

    use_special_tokens=True 

    task = "coupling" #args.task

    #DECISION
    transformer_layers = args.transformer_layers
    decoder_only=True #args.decoder_only 
    inner_dim=args.inner_dim
    heads=args.heads
    dropout = args.dropout
    
    has_masking = False #args.has_masking
    
    VQpath = f"myVQ/kmeans_centers_{vocab_size}_{chunk_duration}s.npy"

    seq2seq=SimpleSeq2SeqModel(pretrained_bb_checkpoint,
                                    bb_type,
                                    dim,
                                    vocab_size,
                                    max_len,
                                    encoder_head,
                                    chunking=pre_post_chunking,
                                    use_special_tokens=use_special_tokens,
                                    task=task,
                                    condense_type=condense_type,
                                    has_masking=has_masking,
                                    freeze_backbone=freeze_backbone,
                                    learnable_codebook=learnable_codebook,
                                    restart_codebook=restart_codebook,
                                    transformer_layers=transformer_layers,
                                    dropout=dropout,
                                    decoder_only=decoder_only,
                                    inner_dim=inner_dim,
                                    heads=heads,
                                    special_vq=args.special_vq,
                                    chunk_size=chunk_duration,
                                    data = args.data,
                                    VQpath = VQpath,
                                    relative_pe=args.relative_pe
                                    )
    return seq2seq

def build_trainer(model, args):
    lr = args.learning_rate
    lr_bb = lr if args.learning_rate_backbone == -1 else args.learning_rate_backbone
    weight_decay= args.weight_decay 
    betas=(0.9, 0.999) #default betas for Adam
    
    k = args.k
    if k>=1:
        k=int(k)
        
    
    criterion = torch.nn.functional.cross_entropy #torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    model = model.to(DEVICE)
    
    optimizer = [torch.optim.Adam(model.parameters(),lr=lr,betas=betas,weight_decay=weight_decay)]
    
    run_id = f"model_{args.chunk_duration}s_A{args.vocab_size}"
    new_run_id=run_id
    i=1
    while os.path.exists(f"runs/{new_run_id}.pt"): #find new name if not resume training
        new_run_id=run_id+f'_{i}'
        i+=1
    run_id=new_run_id
    
    trainer = Seq2SeqTrainer(model,0, criterion, optimizer, run_id,
                            segmentation=args.segmentation,
                            k = k,
                            chunk_size=args.chunk_duration, #PROBLEME VIENT D'ICI ON UTILISE CHUNK SIZE DANS TRAINER ET CA ECRASE LA VALEUR AU PROCHAIN CKP
                            track_size=args.track_duration,
                            scheduled_sampling = args.scheduled_sampling,
                            scheduler_alpha=args.scheduler_alpha,
                            seq_nll_loss=args.seq_nll_loss)
    
    return trainer

def argparser():
    parser = ArgumentParser()
    
    parser.add_argument("--track1", type = str, help = "Path to the input (guide) audio file") #str type because Dataset class still needs update to enable Path type...
    parser.add_argument("--track2", type = str, help = "Path to the output (target) audio file")
    parser.add_argument("--batch_size", type = int, default=None, help = "batch size : default None. If not specified batch size is computed as to have a batch = the whole track")
    parser.add_argument("--pre_segmentation", type=str, default = "sliding", choices = ["sliding", "uniform"])
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-5)
    parser.add_argument('-decay','--weight_decay',type=float,default=1e-5)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument("--scheduled_sampling", action = 'store_true')
    parser.add_argument("--scheduler_alpha", type=float, default = 2)
    
    return parser

def main(args):
    
    tracks = ... #get list of tracks from args
    #train VQ on input tracks
    trainVQ(args.vocab_size, args.chunk_duration, tracks)
    
    #build model
    model = build_model(args)
    
    #build dataset from input tracks
    #source and output tracks are given as separate arguments.
    #each can be a list of several tracks ? -> if multiple tracks are given, coupled tracks should share the same name
    fetcher = build_coupling_ds([args.track1, args.track2],args.batch_size, args.track_duration, args.chunk_duration,distributed=False)
    
    #build trainer
    trainer = build_trainer(model,args)
    #launch training
    trainer.train(fetcher,fetcher,args.epochs,reg_alpha=0) 

if __name__=='__main__':
    args = argparser().parse_args()
    main(args)