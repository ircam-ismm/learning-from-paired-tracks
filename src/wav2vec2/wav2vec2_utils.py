# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:30:02 2024

@author: balth
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from dataclasses import dataclass
from transformers import(
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
    get_scheduler
    )
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.data.data_utils import compute_mask_indices
from munch import Munch
import math
from tqdm import tqdm
from typing import Union, Tuple

@dataclass 
class DataCollatorForWav2Vec2:
    # TODO : ADAPT THIS CLASS TO ACCEPT BACKBONE ? -> MODIFY BACKBONE TO CALL CORRECT FUNCTIONS WHEN CALLED
    model : Union[Wav2Vec2ForPreTraining , Wav2Vec2Model] #PROBLEM WITH TYPE : IT SHOWS BaseFairseqModel AND NOT WAV2VECMODEL
    feature_extractor : Wav2Vec2FeatureExtractor 
    padding : str = "longest"
    mask_time_prob : float = 0.065
    mask_time_length : int = 10
    return_padding_mask : bool = False
    num_classes : int = 12
    split : str = "train" #choices = [train, test, eval, pre-train, adapt]
    
    def _get_mask_indices(self, shape:Tuple[int, int], padding_mask:torch.Tensor = None):
        
        if isinstance(self.model,Wav2Vec2Model):
            #if fairseq model
            mask_indices=compute_mask_indices(shape,padding_mask=padding_mask, 
                                              mask_prob=self.mask_time_prob, mask_length=self.mask_time_length)
            mask_indices=torch.from_numpy(mask_indices)
        
        elif isinstance(self.model,Wav2Vec2ForPreTraining):
            #if HF model
            mask_indices = _compute_mask_indices(shape, attention_mask=padding_mask,
                                                 mask_prob = self.mask_time_prob,
                                                 mask_length = self.mask_time_length)
            
            #to tensor
            mask_time_indices = torch.tensor(data=mask_time_indices, 
                                            dtype=torch.long)
        
        return mask_indices    
    
    def _get_negative_indices(self, shape, mask_time_indices):
        if isinstance(self.model, Wav2Vec2Model):
            #if fairseq the negative samples are computed in the forward method
            pass
        
        elif isinstance(self.model, Wav2Vec2ForPreTraining):
            #HF model
            sampled_negative_indices = _sample_negative_indices(
                features_shape = shape,
                num_negatives = self.model.config.num_negatives,
                mask_time_indices = mask_time_indices
                )
        
        else :
            raise TypeError(f"Expected model type to be either fairseq's {Wav2Vec2Model} or HF's {Wav2Vec2ForPreTraining} not {type(self.model)}")

        
        return sampled_negative_indices
        
    def __call__(self, data):
        #separate the array from the labels
        if len(data[0])==2: 
            with_chunk_time = False
            raw_chunks, labels = zip(*data)
        else : 
            with_chunk_time = True
            raw_chunks, labels, paths, t0, t1 = zip(*data)
        

        # TODO : maybe implement our own preprocessing class/function
        
        #process chunks -> normalize and pad to longest
        processed_chunks = self.feature_extractor(raw_chunks,
                                                  padding=self.padding,
                                                  sampling_rate=self.feature_extractor.sampling_rate,
                                                  return_attention_mask=self.return_padding_mask,
                                                  return_tensors="pt").input_values
        
        #if padding_mask true have to implement the extraction of the padding mask from preprocessor output
        if self.return_padding_mask :
            raise NotImplementedError("Not implemented padding attention mask option")

        #ATTENTION, IN SOME CASES FEATURE EXTRACTOR OUTPUTS ATTENTION MASKS CORRESPONDING
        #TO THE PADDED SEQUENCES. FOR SOME MODELS THOSE MASKS HAVE TO BE USED TO NOT COMPUTE
        #THE LOSS OVER THOSE PADDED INDEXES. For base model there is 
        #no attention mask returned for padded sequences -> loss computed on whole padded sequence
        #MAYBE TO IMPLEMENT in the future ->depends on feature_extractor "return attention mask" attribute
        
        #get input shape to compute encoded sequence length (output of cnn encoder)
        batch_size, raw_sequence_length = processed_chunks.shape
        device=next(self.model.parameters()).device #get device (raised error for fairseq model)
        
        #both fairseq and hf w2v have the same method name
        encoded_sequence_length = self.model._get_feat_extract_output_lengths(
                                torch.tensor(raw_sequence_length, device=device))
        encoded_sequence_length = int(encoded_sequence_length)
        
        
        #compute masked indices (train and eval)
        #mask_time_indices = _compute_mask_indices(shape=(batch_size, encoded_sequence_length),
        #                                          mask_prob = self.mask_time_prob,
        #                                          mask_length = self.mask_time_length)
        
        mask_time_indices = torch.tensor([])
        if self.split in ["pre-train","eval"]: #dont send mask indices when testing the model, only during (pre) training or evaluation (evaluate contrastive task)
            mask_time_indices = self._get_mask_indices(shape=(batch_size, encoded_sequence_length))
        
        sampled_negative_indices = [] 
        
        
        if self.split=="pre-train":
            sampled_negative_indices = self._get_negative_indices((batch_size, encoded_sequence_length), mask_time_indices)
        
        #to tensor
        sampled_negative_indices = torch.tensor(data=sampled_negative_indices,
                                                        dtype=torch.float)
        
        
        #to tensor
        labels = torch.tensor(data=labels)
        labels_oh = one_hot(labels, num_classes=self.num_classes).to(torch.float)
        
        if with_chunk_time :
            t0 = torch.tensor(t0,dtype=torch.float)
            t1 = torch.tensor(t1,dtype=torch.float)
            #paths = torch.tensor(paths,dtype=str)
        
        else :
            t0 = torch.tensor([],dtype=torch.float)
            t1 = torch.tensor([],dtype=torch.float)
        
        inputs = Munch(x = processed_chunks, 
                       mask_indices = mask_time_indices,
                       negative_indices = sampled_negative_indices,
                       instruments = labels,
                       targets=labels_oh,
                       t0=t0,t1=t1)
        
        return inputs

        
class Fetcher:
    #class handling the iteration over the loader and sending to available device
    def __init__(self, loader):
        self.loader=loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _fetch_inputs(self):
        #method to fetch next set of inputs
        try:
            #try to fectch next inputs
            inputs = next(self.iter_loader)
        except (AttributeError, StopIteration):
            #if self.iter_loader not already instantiated or end of loader
            self.iter_loader = iter(self.loader)
            inputs = next(self.iter_loader)
        
        return inputs
    
    def __next__(self):
        inputs = self._fetch_inputs()

        #pass inputs to cuda and as a Munch
        return Munch({key : item.to(self.device) for key, item in inputs.items()})        



#highly inspired from :
#   https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py
class Wav2Vec2Trainer:
    """
    class handling the wav2vec2 pretraining
    """
    def __init__(self, params : Munch, model : Wav2Vec2ForPreTraining):
        self.params = params #training parameters (e.g. batch_size, optimizer params, lr,...)
        self.model = model #wav2vec2 model to pre-train
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            params = list(self.model.parameters()),
            lr = params.lr,
            betas=[params.adam_beta1,params.adam_beta2],
            eps=params.adam_eps
            )
        
    def get_grad_norm(self, params):
        total_norm=0.0
        for p in params:
            if p.grad is not None:
                param_norm=(p.grad.detach().data).norm(2)
                total_norm += param_norm.item()**2 
        total_norm = total_norm ** 0.5 
        return total_norm
                
    def multiply_grad(self, params, c):
        #scale the gradients by a factor c
        #needed to scale down the gradients by the total number of losses (# of masked indices)
        for p in params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    #if c is a tensor, send to device
                    c = c.to(p.device)
                p.grad.data.mul_(c) #multiplky gradient of parameter by c
                
    def train(self, train_loader : DataLoader, val_loader : DataLoader):
        
        train_fetcher = Fetcher(train_loader)
        val_fetcher = Fetcher(val_loader)
        
        epochs = self.params.epochs
        #the number of iterations in an epoch is reduced by the number of 
        #steps for the gradient accumulation
        num_iter_per_epoch = math.ceil(len(train_loader)/self.params.grad_acc_steps)
        
        #if max train steps is not specified
        if self.params.max_train_steps is None:
            self.params.max_train_steps = epochs * num_iter_per_epoch
        
        #learning rate scheduler
        lr_scheduler = get_scheduler(
            name = self.params.lr_scheduler_type,
            optimizer = self.optimizer,
            num_warmup_steps = self.params.num_warmup_steps,
            num_training_steps = self.params.max_train_steps
            )
        
        #update epochs 
        epochs = math.ceil(self.params.max_train_steps / num_iter_per_epoch)
        
        completed_steps = 0 #total of optimizers steps
        
        progress_bar = tqdm(range(self.params.max_train_steps))
        #train loop
        for epoch in range(epochs):
            self.model.train()
            for step in range(len(train_loader)):
                
                inputs = next(train_fetcher)
                
                num_losses = inputs.mask_indices.sum() #total number of losses
                
                #forward
                outputs = self.model(inputs.x,
                                mask_time_indices=inputs.mask_indices,
                                sampled_negative_indices=inputs.negative_indices)
                
                #loss accumulation
                loss = outputs.loss / self.params.grad_acc_steps
                loss.backward()
                
                #scaled down gradients (average over all losses)
                self.multiply_grad(self.model.parameters(), 1/num_losses)
                
                #update step every grad accumulation steps
                if (step+1)%self.params.grad_acc_steps == 0 or (step+1) == len(train_loader)-1:
                    
                    #backward pass
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    #scheduler step
                    lr_scheduler.step()
                    
                    #gumbel softmax temperature update
                    gumbel_temperature = max(
                        self.params.gumbel_max_temp*self.params.gumbel_temp_decay**completed_steps,
                        self.params.gumbel_min_temp
                        )
                    
                    #set gumbel temperature in model
                    self.model.set_gumbel_temperature(gumbel_temperature)
                    
                    completed_steps+=1
                    progress_bar.update(1) #update progress bar
                    
            
            #validation
            self.model.eval()
            
            #logs
            val_logs = {
                "val_loss":0,
                "val_contrastive_loss":0,
                "val_diversity_loss":0,
                "val_num_losses":0
                }
            
            print("\ncomputing validation loss")
            for _ in range(len(val_loader)):
                val_inputs = next(val_fetcher)
                with torch.no_grad():
                    val_outputs = self.model(val_inputs.x,
                                             mask_time_indices=val_inputs.mask_indices,
                                             sampled_negative_indices=val_inputs.negative_indices)
                
                val_logs["val_loss"]+=val_outputs.loss
                val_logs["val_contrastive_loss"]+=val_outputs.contrastive_loss
                val_logs["val_diversity_loss"]+=val_outputs.diversity_loss
                val_logs["val_num_losses"]+=val_inputs.mask_indices.sum()
            
            val_logs = {k:v/val_logs["val_num_losses"] for k,v in val_logs.items()}
            
            log = "\n"
            for k, v in val_logs.items():
                if k=="val_num_losses" : continue
                log += f"{k} = {v.item()} |"
            
            #write log
            progress_bar.write(log)
            
            #save model if validation loss if better
            #TODO 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
