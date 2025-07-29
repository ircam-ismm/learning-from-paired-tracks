import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from MusicDataset.MusicDataset_v2 import Fetcher
from typing import Callable, Tuple
from tqdm import tqdm
from architecture.Seq2Seq import Seq2SeqCoupling,Seq2SeqBase
from architecture.Model import load_model_checkpoint,myDDP
from abc import ABC, abstractmethod
from utils.utils import *
from utils.metrics import compute_accuracy
import matplotlib.pyplot as plt
import numpy as np
import time, os
import optuna
from munch import Munch 
from typing import List,Union

# TODO : IMPLEMENT ABSTRACT BASE TRAINER CLASS FOR GENRAL PURPOSES
class Trainer(ABC):
    pass


#CHANGE TO ENTROPY
def compute_codebook_usage(idxs,vocab_size,pad_idx):
    idxs_nopad = [idx.numpy(force=True) for idx in idxs if idx!=pad_idx]
    
    counts = np.bincount(idxs_nopad,minlength=vocab_size)
    usage = np.count_nonzero(counts)/len(counts)
    
    return usage

class Seq2SeqTrainer(nn.Module):
    def __init__(self, 
                 model : Seq2SeqBase, 
                 gpu_id : int,
                 criterion : Callable,
                 optimizer : List[Optimizer],
                 trainer_name : str,
                 segmentation : str,
                 save_ckp : bool = True,
                 grad_accum_steps : int = 1,
                 codebook_loss_weight : float = 1.,
                 k : int = None,
                 chunk_size : float = 0.5,
                 track_size : float = 30,
                 #resume_epoch : int =0,
                 resume_ckp : str = "",
                 init_sample_temperature : float = 2.,
                 min_temperature : float = 0.5,
                 with_decay : bool = False,
                 weighed_crossentropy : bool = False,
                 scheduled_sampling : bool = True,
                 scheduler_alpha : float = 2,
                 seq_nll_loss : bool = False):
        
        super().__init__()
        self.model=model
        self.gpu_id = gpu_id
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        #add some options for checkpoint saving etc
        self.trainer_name = trainer_name
        self.segmentation = segmentation
        self.save_ckp=save_ckp
        #self.resume_epoch = resume_epoch
        self.resume_ckp=resume_ckp
        
        #VQ parameters (not used if kmeans vq)
        self.codebook_loss_alpha = codebook_loss_weight
        assert init_sample_temperature>=min_temperature, "init sample temp should be higher than min temperature"
        self.codebook_sample_temperature = init_sample_temperature
        self.min_temperature = min_temperature
        self.with_decay = with_decay
        
        #topK prediction
        if k == None : k = min(max(int(0.1*self.model.vocab_size),5),100) #10% du vocab - seuil a 5 et 100
        self.k = k
        self.chunk_size=chunk_size
        self.track_size=track_size
        
        self.weighed_crossentropy = weighed_crossentropy
        self.weights = None
        
        self.scheduled_sampling = scheduled_sampling
        self.scheduler_alpha = scheduler_alpha
        
        self.seq_nll_loss = seq_nll_loss
        
        
    def save_checkpoint(self, ckp_name : str, perfs : dict):
        if not any(ckp_name.endswith(ext) for ext in (".pt",".pth")):
            raise ValueError(f"checkpoint filename must end with .pt or .pth")
        
        model = self.model.module if isinstance(self.model,DDP) else self.model #if ddp
        
        model_params = {"backbone_type":self.model.encoder.encoder.type,
                        "freeze_backbone":self.model.encoder.encoder.frozen,
                        "dim":model.dim,
                        "pre_post_chunking":model.encoder.chunking_pre_post_encoding,
                        "vocab_size":self.model.codebook_size,
                        "learnable_codebook" : self.model.encoder.quantizer.learnable_codebook,
                        "special_vq" : self.model.encoder.quantizer.is_special,
                        "vq_data" : self.model.encoder.quantizer.data,
                        "chunk_size" : self.chunk_size,
                        "tracks_size" : self.track_size,
                        "max_len":self.model.pe.size,
                        "encoder_head":self.model.encoder.head_module,
                        "condense_type":self.model.encoder.condense_type,
                        "use_special_tokens":self.model.use_special_tokens,
                        "has_masking" : self.model.has_masking,
                        "mask_prob" : self.train_fetcher.loader.collate_fn.mask_prob,
                        "mask_len" : self.train_fetcher.loader.collate_fn.mask_len,
                        "task" : "coupling" if type(model)==Seq2SeqCoupling else "completion",
                        "decoder_only":self.model.decision.decoder_only,
                        "transformer_layers":self.model.decision.layers,
                        "relative_pe" : self.model.decision.relative_pe,
                        "dropout":self.model.decision.dropout,
                        "inner_dim":self.model.decision.inner_dim,
                        "heads":self.model.decision.heads,
                        "norm_first":self.model.decision.norm_first,
                        "segmentation": self.segmentation,
                        "top-K" : self.k,
                        "run_id" : self.trainer_name
                        
                  } 
        state_dict = model.state_dict() #self.model.state_dict() if isinstance(self.model,Seq2SeqBase) else self.model.module.state_dict() #if DDP model.module
        optim_state_dict = [optim.state_dict() for optim in self.optimizer]
        torch.save({
            "model_class":model.__class__,
            "state_dict":state_dict,
            "optimizer":optim_state_dict,
            "model_params":model_params,
            "perfs" : perfs
            },"runs/coupling/"+ckp_name)
    
    #TODO : FIND HOW TO RELOAD CHECKPOINT DURING DDP TRAINING
    # check : https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
    def load_checkpoint(self,checkpoint_name):
        print("ici avec rank :",self.gpu_id)
        #torch.distributed.init_process_group("nccl",rank=self.gpu_id,world_size = torch.distributed.get_world_size())
        # torch.cuda.set_device(self.gpu_id)
        # torch.cuda.empty_cache()
        #load ceckpoint
        model, params, optim_state_dict = load_model_checkpoint(checkpoint_name)
        #send to rank
        # device=torch.cuda.current_device()
        # print(device)
        model = model.to(self.gpu_id)
        #wrap inside DDP
        model = myDDP(model, device_ids=[int(self.gpu_id)],
                      find_unused_parameters= not params['freeze_backbone'] or params['learnable_codebook']) 
        
        print("la")
        #assign model to model attribute of trainer 
        self.model = model
        
        #load optimizer
        if type(optim_state_dict) == list :
            for i,optim in enumerate(optim_state_dict):
                self.optimizer[i].load_state_dict(optim)
        
        else : self.optimizer.load_state_dict(optim_state_dict)
        
        print("avant barier()")
        torch.distributed.barrier()
        print("apres barrier")
    
    def _forward(self,inputs:Munch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        if type(model)==Seq2SeqCoupling:
            src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, scheduled_prob = inputs.values()
            #compute output
            logits, tgt, tgt_idx, codebook_loss = self.model.forward(src, tgt, src_pad_mask, tgt_pad_mask,
                                                        sample_codebook_temp=self.codebook_sample_temperature,
                                                        mask_time_indices = src_mask_indices,
                                                        scheduled_sampling_prob=scheduled_prob)
            
        elif type(model)==Seq2SeqBase: #for autocompletion
            src, src_pad_mask, src_mask_indices, label, scheduled_prob = inputs.values() 
            #compute output
            logits, tgt, tgt_idx, codebook_loss = self.model.forward(src, src_pad_mask, 
                                                             sample_codebook_temp=self.codebook_sample_temperature,
                                                             mask_time_indices = src_mask_indices,
                                                             )
            
        return logits,tgt,tgt_idx,codebook_loss

    @torch.no_grad    
    def evaluate(self,eval_fetcher, reg_alpha, scheduled_prob):
        prYellow("Evaluation...")
        loss=0
        loss_ce=0
        loss_entropy=0
        loss_commit=0
        acc=0
        cb_usage=0
        cb_usage_gt=0
        self.model.eval()
        for _ in range(len(eval_fetcher)):
            inputs = next(eval_fetcher)
            inputs['scheduled_prob']=scheduled_prob
            
            logits,tgt,tgt_idx,codebook_loss = self._forward(inputs)
            
            tgt_out = tgt_idx[:,1:] #ground truth
            
            loss_, separate_losses = self._compute_loss(logits, tgt_out, reg_alpha, codebook_loss)
            loss_ce_ , loss_entropy_, loss_commit_ = separate_losses
            
            loss+=loss_
            loss_ce+=loss_ce_
            loss_entropy+=loss_entropy_
            loss_commit+=loss_commit_
            
            #loss_ce=self.criterion(logits.reshape(-1,logits.size(-1)), tgt_out.reshape(-1)).item() #reshqaped as (B*T,vocab_size) and (B*T,)
            
            #loss += loss_ce + self.codebook_loss_alpha*codebook_loss
            
            #topK search
            preds = predict_topK_P(self.k,logits,tgt_out)
            
            acc += compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=self.model.special_tokens_idx["pad"])
            
            cb_usage += compute_codebook_usage(preds,self.model.vocab_size,pad_idx=self.model.special_tokens_idx["pad"])
            cb_usage_gt += compute_codebook_usage(tgt_out.reshape(-1),self.model.vocab_size,pad_idx=self.model.special_tokens_idx["pad"])
        
        loss=loss/len(eval_fetcher)
        loss_ce=loss_ce/len(eval_fetcher)
        loss_entropy=loss_entropy/len(eval_fetcher)
        loss_commit=loss_commit/len(eval_fetcher)
        separate_losses = (loss_ce, loss_entropy, loss_commit)
        acc/=len(eval_fetcher)
        cb_usage/=len(eval_fetcher)
        cb_usage_gt/=len(eval_fetcher)
        print("Codebook usage:",cb_usage)
        print("Codebook usage GT:",cb_usage_gt)
        
        return loss, acc, cb_usage, separate_losses
    
    def plot_loss(self, epoch, train_losses, val_losses, train_acc, val_acc, name):
        fig, ax1 = plt.subplots(figsize=(10,10),dpi=150)
        ax2=ax1.twinx()
        epochs = range(1,epoch+2)
        #print(len(epochs),len(train_losses), len(val_losses), len(train_acc), len(val_acc))
        #plt.figure(figsize=(10,10),dpi=150)
        ax1.plot(epochs,train_losses,label="train loss", color="tab:blue")
        ax2.plot(epochs,train_acc,"--",label="train accuracy",color="tab:green")
        if len(val_losses) != 0:
            ax1.plot(epochs,val_losses,label="val loss", color="tab:orange")
            ax2.plot(epochs, val_acc, "--",label="val accuracy", color="tab:red")
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")

        
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax2.grid()
        fig.savefig(f"runs/coupling/{name}_{self.trainer_name}.png")
        #fig.tight_layout()
        #fig.show()
        plt.close()
        
    def plot_loss_separate(self, epoch, train_losses_ce, train_losses_entropy, val_losses_ce, val_losses_entropy, train_acc, val_acc):
        fig, ax1 = plt.subplots(figsize=(10,10),dpi=150)
        ax2=ax1.twinx()
        epochs = range(1,epoch+2)
        #plt.figure(figsize=(10,10),dpi=150)
        ax1.plot(epochs,train_losses_ce,label="train loss ce", color="tab:blue")
        ax1.plot(epochs,train_losses_entropy,label="train loss entropy")
        ax2.plot(epochs,train_acc,"--",label="train accuracy",color="tab:green")
        if len(val_losses_ce) != 0:
            ax1.plot(epochs,val_losses_ce,label="val loss ce", color="tab:orange")
            ax1.plot(epochs,val_losses_entropy,label="val loss entropy")
            ax2.plot(epochs, val_acc, "--",label="val accuracy", color="tab:red")
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Cross Entropy / Entropy")
        ax2.set_ylabel("Accuracy")
        ax2.grid()
        fig.savefig(f"runs/coupling/Loss_{self.trainer_name}_separate.png")
    
    def sequence_nll_loss(self, logits : torch.Tensor, targets : torch.Tensor, pad_index: torch.Tensor) -> torch.Tensor:
        """
        Computes sequence-level Negative Log-Likelihood (NLL) loss.

        Args:
        - logits: (batch_size, seq_len, vocab_size) -> Model output logits.
        - targets: (batch_size, seq_len) -> Ground truth token indices.
        - padding_mask: (batch_size, seq_len) -> Boolean mask (True for padding).

        Returns:
        - loss: A scalar representing the sequence loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        #construct padding mask from pad_index and targets
        padding_mask = torch.where(targets==pad_index,torch.tensor(True,device=targets.device),torch.tensor(False,device=targets.device))
        
        # Apply padding mask
        nll_loss = nll_loss * (~padding_mask)

        # Sum over sequence and normalize
        seq_loss = nll_loss.sum(dim=1)  # Sum across sequence
        seq_lengths = (~padding_mask).sum(dim=1)  # Count valid tokens
        loss_per_seq = seq_loss / seq_lengths.clamp(min=1)  # Avoid division by zero

        return loss_per_seq.mean()
    
    #TODO : ADD reg_alpha to class attributes ?
    def _compute_loss(self,logits,tgt_out,reg_alpha,codebook_loss) -> torch.Tensor :
        
        y = logits.reshape(-1,logits.size(-1))
        gt = tgt_out.reshape(-1)
        
        pad_idx = self.model.special_tokens_idx["pad"] if self.model.use_special_tokens else -100
        
        if self.seq_nll_loss:
            loss_ce = self.sequence_nll_loss(logits, tgt_out, pad_idx)
        
        else :
            loss_ce = self.criterion(y, gt,ignore_index = pad_idx, weight = self.weights, label_smoothing=0.1) #reshqaped as (B*T,vocab_size) and (B*T,)
        
        loss_commit = self.codebook_loss_alpha*codebook_loss
        
        #add codebook diversity loss ?
        
        #entropy regularization --> maximize entropy = use most of vocabulary
        probs = F.softmax(logits,dim=-1)
        entropy = -1.*(torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean())
        loss_entropy = reg_alpha*entropy
        #loss_entropy=0
        
        #totasl loss = crossentropy + codebook loss + (-entropy). - entropy because we want to maximize and loss decreases when entropy rises
        loss = (loss_ce + loss_commit - loss_entropy)/self.grad_accum_steps 
        
        return loss, (loss_ce, loss_entropy, loss_commit)
    
    def _run_batch(self,train_fetcher,reg_alpha,step,trial,scheduled_prob):    
        
        inputs = next(train_fetcher)
        inputs['scheduled_prob']=scheduled_prob
       
        logits,tgt,tgt_idx,codebook_loss = self._forward(inputs)
                
        #zero grad
        for optim in self.optimizer : optim.zero_grad()
        
        tgt_out = tgt_idx[:,1:] #ground truth outputs are the token indexes shifted to the right
        
        loss, separate_losses = self._compute_loss(logits, tgt_out, reg_alpha, codebook_loss)
        
        preds = predict_topK_P(self.k,logits,tgt_out) 
        
        acc = compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=self.model.special_tokens_idx["pad"])
        
        preds = preds.reshape(logits.shape[:-1]) #reshape to B,chunks
        
        if step%5==0 and trial==None:
            if self.gpu_id==0:
                for i in range(min(3,len(preds))):
                    prYellow(f"Pred {preds[i].numpy(force=True)}")
                    prYellow(f"GT {tgt_out[i].numpy(force=True)}")
                # prYellow(f"Pred {preds[1].numpy(force=True)}")
                # prYellow(f"GT {tgt_out[1].numpy(force=True)}")
                # prYellow(f"Pred {preds[2].numpy(force=True)}")
                # prYellow(f"GT {tgt_out[2].numpy(force=True)}")
                #prRed(f"Pred {torch.argmax(logits[2],-1).numpy(force=True)}")
                prRed(f"{torch.topk(logits.softmax(-1)[i,:5],k=5)}") #show topk probs of 10 first tokens
                prYellow(loss.item())
                
                
        
        params = self.model.parameters() if not isinstance(self.model,DDP) else self.model.module.parameters()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0)
        if step%self.grad_accum_steps == 0 or step == len(train_fetcher):
            for optim in self.optimizer : optim.step()
        
        return loss, acc, separate_losses
    
    @torch.no_grad
    def _compute_class_weights(self,train_fetcher : Fetcher):
        prYellow("Computing class weights...")
        density=torch.zeros(self.model.vocab_size, device=self.model.device)
        for _ in range(len(train_fetcher)):
            inputs = next(train_fetcher)
            src, src_pad_masks = inputs.src, inputs.src_padding_masks
            src, src_idx, _ = self.model.encode(src,src_pad_masks)
            density += torch.bincount(src_idx.reshape(-1),minlength=self.model.vocab_size)
            
        density = density/sum(density)
        min_nonzero = density[density > 0].min()
        density = torch.where(density == 0, min_nonzero, density)   
        weights = 1/density
        weights = weights/sum(weights) #normalize
        return weights
                
    def train(self, train_fetcher, val_fetcher,epochs, evaluate=True,reg_alpha=0.1, trial : optuna.trial.Trial = None, save_every : int = 1):
        if evaluate :
            assert val_fetcher != None, "To evaluate the model a validation set is needed."
        
        train_iter=epochs*len(train_fetcher) #total iterations
        
        train_losses=[]
        train_losses_ce = []
        train_losses_entropy = []
        train_losses_commit=[]
        val_losses=[]
        val_losses_ce = []
        val_losses_entropy = []
        val_losses_commit=[]
        train_accs = []
        val_accs=[]
        resume_epoch=0
        if self.resume_ckp!="":
            try :
                # path=os.path.abspath(__file__)
                # dir = os.path.dirname(path)
                # print("Current dir :", dir)
                # d=np.load(f"{dir}/runs/coupling/eval_{self.trainer_name}.npy",allow_pickle=True).item()
                ckp = torch.load(self.resume_ckp,map_location="cpu") 
                perfs = ckp["perfs"]
                train_losses=perfs['train_loss']
                train_losses_ce=perfs["train_loss_ce"]
                train_losses_commit=perfs["train_loss_commit"]
                train_losses_entropy=perfs["train_loss_entropy"]
                train_accs = perfs['train_acc']
                
                val_losses=perfs['test_loss']
                val_losses_ce=perfs["test_loss_ce"]
                val_losses_commit=perfs["test_loss_commit"]
                val_losses_entropy=perfs["test_loss_entropy"]
                val_accs = perfs['test_acc']
                
                resume_epoch = len(train_losses) #assign value to avoir error during plot etc
                
                
                #del ckp #delete checkpoint after loading performances
                
            except KeyError as e:
                print(f"Problem loading performances form checkpoint {self.resume_ckp}")
                print("Trying to load from eval*.npy file")
                ckp_path = Path(self.resume_ckp)
                path = f"{ckp_path.parent}/eval_{ckp_path.stem}.npy"
                perfs = np.load(path,allow_pickle=True).item()
                                
                train_losses=perfs['train_loss']
                #old versions dont save all losses separately so we copy them
                train_losses_ce=perfs["train_loss"].copy()
                train_losses_commit=perfs["train_loss"].copy()
                train_losses_entropy=perfs["train_loss"].copy()
                train_accs = perfs['train_acc']
                
                val_losses=perfs['test_loss']
                #old versions dont save all losses separately so we copy them
                val_losses_ce=perfs["test_loss"].copy()
                val_losses_commit=perfs["test_loss"].copy()
                val_losses_entropy=perfs["test_loss"].copy()
                val_accs = perfs['test_acc']
                
                resume_epoch = len(train_losses) #assign value to avoir error during plot etc
                #print("Resume epoch =", resume_epoch)
                
                
        
        best_loss = float('inf') if len(val_losses)==0 else min(val_losses)
        best_acc = 0 if len(val_accs)==0 else max(val_accs)
        best_codebook_usage=0
        
        init_temperature = self.codebook_sample_temperature
        
        self.train_fetcher = train_fetcher #to get params for saving
        
        iter_count=0
        if self.gpu_id==0:
            progress_bar = tqdm(total=train_iter,initial=resume_epoch*len(train_fetcher))
        

        if self.weighed_crossentropy:
            self.weights = self._compute_class_weights(train_fetcher)
            print(self.weights)
        
        for epoch in range(resume_epoch,epochs):
            print("Current epoch =", epoch+1)
            train_loss=0
            train_loss_ce=0
            train_loss_entropy=0
            train_loss_commit=0
            train_acc=0
            val_loss=0
            self.model.train()
            
            #with ddp we need to set epoch on sampler before creating dataloader iterator (done in fetcher when restarting loader)
            try :
                train_fetcher.loader.sampler.set_epoch(epoch) 
            except:
                pass
            
            #shceduled sampling probability
            scheduled_prob = None
            if self.scheduled_sampling:
                k = -np.log(1e-9) / self.scheduler_alpha
                scheduled_prob = min(0.95, 1 - np.exp(-epoch*k/epochs)) #exponential decay
                #scheduled_prob = 1/(1+np.exp(-(epoch-epochs/2)/self.scheduler_alpha))
                print(f"schdeduled sampling prob at epoch {epoch+1} =",scheduled_prob)
                
                if scheduled_prob==0: scheduled_prob=None  #dont do scheduled sampling if none is needed
                
            
            for step in range(len(train_fetcher)):
                if self.gpu_id==0:
                    progress_bar.update(1) #update progress bar
                iter_count+=1
                
                #get inputs and targets
                #inputs = next(train_fetcher)
                
                loss, acc, separate_losses = self._run_batch(train_fetcher,reg_alpha,step,trial,scheduled_prob)
                loss_ce, loss_entropy, loss_commit = separate_losses
                
                #print(separate_losses)
                
                train_loss+=loss.item()
                train_loss_ce+=loss_ce.item()
                train_loss_entropy+=loss_entropy.item()
                train_loss_commit+=loss_commit.item()
                
                train_acc+=acc
                
                #temperature annealing
                if self.with_decay:
                    new_temp = init_temperature - (init_temperature-self.min_temperature)/(train_iter) * iter_count
                    self.codebook_sample_temperature = min(new_temp,self.min_temperature)
                
                
            train_loss = train_loss/len(train_fetcher)
            train_loss_ce = train_loss_ce/len(train_fetcher)
            train_loss_commit = train_loss_commit/len(train_fetcher)
            train_loss_entropy = train_loss_entropy/len(train_fetcher)
            
            train_losses.append(train_loss)
            #print("after appending new train_loss:",len(train_losses),len(train_losses_ce),len(train_losses_commit),len(train_losses_entropy))
            train_losses_ce.append(train_loss_ce)
            #print("after appending new ce_loss:",len(train_losses),len(train_losses_ce),len(train_losses_commit),len(train_losses_entropy))
            train_losses_entropy.append(train_loss_entropy)
            #print("after appending new entropy_loss:",len(train_losses),len(train_losses_ce),len(train_losses_commit),len(train_losses_entropy))
            train_losses_commit.append(train_loss_commit)
            
            #print("after appending new commit_loss:",len(train_losses),len(train_losses_ce),len(train_losses_commit),len(train_losses_entropy))
                            
            train_acc/=len(train_fetcher)
            train_accs.append(train_acc)
            
            prGreen(f"Training loss at epoch {epoch+1}/{epochs} : {train_loss}. Accuracy = {train_acc}")
            #val loss
            if evaluate:
                val_loss, val_acc, codebook_usage, separate_losses_val = self.evaluate(val_fetcher, reg_alpha, scheduled_prob)
                val_loss = val_loss.item()
                loss_ce_val, loss_entropy_val, loss_commit_val = separate_losses_val
                loss_ce_val = loss_ce_val.item() 
                loss_entropy_val = loss_entropy_val.item() 
                loss_commit_val = loss_commit_val.item() 
                prGreen(f"Validation loss at epoch {epoch+1}/{epochs} : {val_loss}. Accuracy = {val_acc}")
            
            else : 
                val_loss = train_loss #for checkpooint saving
                val_acc = train_acc
                codebook_usage=best_codebook_usage
                loss_ce_val, loss_entropy_val, loss_commit_val = train_loss_ce, train_loss_entropy, train_loss_commit
            
            
            
            val_losses.append(val_loss) 
            val_losses_ce.append(loss_ce_val)
            val_losses_entropy.append(loss_entropy_val)
            val_losses_commit.append(loss_commit_val)
            val_accs.append(val_acc) 
            
            #print("after appending new loss (val):",len(train_losses),len(train_losses_ce),len(train_losses_commit),len(train_losses_entropy))
            
                        
            if epoch>0 and trial==None:   
                if self.gpu_id==0:
                    self.plot_loss(epoch,train_losses, val_losses, train_accs, val_accs, "total_loss")
                    self.plot_loss(epoch,train_losses_ce, val_losses_ce, train_accs, val_accs, "crossentropy_loss")
                    self.plot_loss(epoch,train_losses_entropy, val_losses_entropy, train_accs, val_accs, "entropy_loss")
                    self.plot_loss(epoch,train_losses_commit, val_losses_commit, train_accs, val_accs, "commit_loss")
                    #self.plot_loss_separate(epoch,train_losses_ce,train_losses_entropy, val_losses_ce, val_losses_entropy, train_accs, val_accs)
                    
            if True:#val_loss<best_loss:#val_loss<best_loss:
                best_loss=val_loss
                best_codebook_usage = codebook_usage #like so they are not totally decorrelated during optim ?
                best_acc = val_acc
                if self.save_ckp:
                    if self.gpu_id==0 and (epoch+1)%save_every==0: #only save rank 0 model
                        prGreen(f"Saving checkpoint : val_loss = {val_loss}, val_acc = {val_acc}")
                        #print('save ckp')
                        perfs = {
                            "train_loss":train_losses,
                            "train_loss_ce":train_losses_ce,
                            "train_loss_entropy":train_losses_entropy,
                            "train_loss_commit":train_losses_commit,
                            "test_loss":val_losses,
                            "test_loss_ce":val_losses_ce,
                            "test_loss_entropy":val_losses_entropy,
                            "test_loss_commit":val_losses_commit,
                            "train_acc":train_accs,
                            "test_acc":val_accs
                            }
                        self.save_checkpoint(self.trainer_name+".pt", perfs) #save ckp as run name and add .pt extension
                    
                    # # Synchronize across all ranks
                    # The above code is using the `torch.distributed.barrier()` function to
                    # synchronize all processes in a distributed setting. It ensures that all
                    # processes reach a specific point in the code before any of them can proceed
                    # further. In this case, it is used to ensure that rank 0 finishes saving before
                    # the other processes can proceed.
                    
                    # torch.distributed.barrier()  # Ensure rank 0 finishes saving before others proceed

                    # # # Reload the checkpoint on non-zero ranks
                    # if self.gpu_id != 0:
                    #     checkpoint_name = f"runs/coupling/{self.trainer_name}.pt"
                    #     self.load_checkpoint(checkpoint_name)
                
            
            # if self.gpu_id==0: #check rank
            #     np.save(f"runs/coupling/eval_{self.trainer_name}.npy",{"train_loss":train_losses,"test_loss":val_losses,"train_acc":train_accs,"test_acc":val_accs},allow_pickle=True)
        
        if trial!=None:
            return best_loss,best_codebook_usage
            
                