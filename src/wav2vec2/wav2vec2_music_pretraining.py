# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:36:56 2024

@author: balth
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining, AutoConfig
from MusicDataset.MusicDataset_v2 import MusicContainer, Fetcher
from wav2vec2.wav2vec2_utils import DataCollatorForWav2Vec2, Wav2Vec2Trainer
from munch import Munch

#%%
checkpoint="facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
model = Wav2Vec2ForPreTraining.from_pretrained(checkpoint)

#no pretraining on speech
# config = AutoConfig.from_pretrained(checkpoint)
# model = Wav2Vec2ForPreTraining(config) 
#%%

root = "../data/train"

train_dataset = MusicContainer(root, max_duration=10.0, 
                         sampling_rate=feature_extractor.sampling_rate,
                         segmentation_strategy="uniform")

val_root="../data/val"
val_dataset = MusicContainer(val_root, max_duration=10.0, 
                         sampling_rate=feature_extractor.sampling_rate,
                         segmentation_strategy="uniform")

#%%

collate_fn=DataCollatorForWav2Vec2(model, feature_extractor, split="train")

batch_size=8
train_dataloader = DataLoader(train_dataset, 
                              batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

val_dataloader = DataLoader(val_dataset, 
                              batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)


#%%
params = Munch(
    lr = 5e-5,
    epochs=3,
    max_train_steps=None,
    grad_acc_steps = 1,
    lr_scheduler_type="linear",
    num_warmup_steps = 0,
    gumbel_max_temp=2.0,
    gumbel_min_temp=0.5,
    gumbel_temp_decay=0.999995,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8  
    )

trainer = Wav2Vec2Trainer(params, model)
    
    
#%%
trainer.train(train_dataloader, val_dataloader)














