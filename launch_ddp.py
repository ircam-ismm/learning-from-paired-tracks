import subprocess
import os
from argparse import ArgumentParser

def train_parser():
    parser = ArgumentParser(description="Coupling training script wit DDP")

    parser.add_argument('--device_ids',default=None)
    parser.add_argument('--num_devices',type=int,default=2)
    parser.add_argument('-cd','--chunk_duration', type=float, default = 0.5)
    parser.add_argument('-td','--track_duration', type = float, default=15.)
    parser.add_argument('-seg',"--segmentation", type=str, default="uniform")
    parser.add_argument('-p_seg',"--pre_segmentation", type=str, default="sliding")
    parser.add_argument('--pre_post_chunking',type = str, choices=['pre','post'], default = 'post')
    parser.add_argument('-dir','--direction',type=str,choices=["stem","mix"],default="stem")
    parser.add_argument('--dim', type = int, choices=[256,768], default=768)
    parser.add_argument('--freeze_backbone',action='store_true')
    #prRed("TRAINIGN BACKBONE")
    parser.add_argument('-vocab','--vocab_size',type=int,choices=[16,32,64,128,256,512,1024,-1],default=-1)
    parser.add_argument("--special_vq", action='store_true')
    parser.add_argument('--learnable_cb',action='store_true')
    parser.add_argument('-restart','--restart_codebook',action='store_true')
    parser.add_argument('--codebook_loss_weight',type=float,default=0.) #++ encoding et embeddings vont se rapprocher vite -> risque de collapse
    parser.add_argument('-head','--encoder_head',type=str,choices=['mean','attention'],default='mean')
    parser.add_argument('-condense','--condense_type',choices=['mask','weighed','none'],default='none')
    parser.add_argument('-layers','--transformer_layers',type=int,default=6)
    parser.add_argument("--relative_pe",action="store_true")
    parser.add_argument("--has_masking",action = 'store_true')
    parser.add_argument("--mask_prob",type=float,default=0.)
    parser.add_argument("--mask_len",type=int,default=0)
    parser.add_argument('--inner_dim',type=int,default=2048)
    parser.add_argument('--heads',type=int,default=12)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--task',type=str,choices=['completion','coupling'],default='coupling')
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-6)
    parser.add_argument('-lr_bb','--learning_rate_backbone',type=float,default=-1)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument("--scheduled_sampling", action = 'store_true')
    parser.add_argument("--scheduler_alpha", type=float, default = 4)
    parser.add_argument('--batch_size',type=int,default=24)
    parser.add_argument('-decay','--weight_decay',type=float,default=1e-5)
    parser.add_argument('--reg_alpha',type=float,default=0.) # ++ -> probs seront uniforme : compromis entre variete et confidence
    parser.add_argument('--grad_accum',type=int,default=1)
    parser.add_argument('--weighed_crossentropy',action='store_true')
    parser.add_argument("--seq_nll_loss",action = "store_true")
    parser.add_argument('--k',type=float,default=1)
    parser.add_argument('--run_id',type=str)
    parser.add_argument('--train_subset',action='store_true') #to do small trainings to find good parameters
    parser.add_argument('--data',choices=['all','canonne','moises','None'])
    parser.add_argument('--resume_ckp',default='')
    #parser.add_argument('--resume_epoch',type=int,default=0)
    
    
    return parser#.parse_args()

#PROBLEM SENDING ARGS ACCROSS PROCESSESß
if __name__=='__main__':
    from utils.utils import lock_gpu,prRed
    
    parser = train_parser()
    args=parser.parse_args()
    
    devices,ids = lock_gpu(args.num_devices)
    
    prRed(f"LOCKED GPUS WITH ID'S:{ids}")
    device_ids=os.environ["CUDA_VISIBLE_DEVICES"]
    path=os.path.abspath(__file__)
    dir = os.path.dirname(path)
    
    args.device_ids=device_ids
    # Convert Namespace to list of strings
    args_list = []
    for arg, value in vars(args).items():
        if isinstance(value, bool):
            if value:
                args_list.append(f'--{arg}')
        elif isinstance(value,list):
            s=f'--{arg}'
            args_list.append(s)
            s=','.join(str(v) for v in value)            
            args_list.append(s)
        else:
            args_list.append(f'--{arg}')
            args_list.append(f'{str(value)}')
    
    #print(['python', f"{dir}/distributed_training.py"]+args_list)
    
    subprocess.run(['python', f"{dir}/distributed_training.py"]+args_list)
    #subprocess.run(['python', f"{dir}/distributed_training.py"])