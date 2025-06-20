import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
#from vector_quantize_pytorch import EuclideanCodebook, CosineSimCodebook, gumbel_sample, identity
from torch import einsum
from einops import rearrange
from typing import Union, Tuple

#vector quantiozer from pre-computed centers with kmeans algorithm
class KmeansQuantizer(nn.Module):
    def __init__(self,centers : Union[np.ndarray,torch.Tensor], learnable_codebook : bool, dim : int = 768, restart : bool = False, is_special : bool = True, data : str = None):
        super().__init__()
        
        if isinstance(centers,np.ndarray) : centers = torch.from_numpy(centers)
    
        self.codebook_size=len(centers)
        self.centers=centers 
        
        if dim==centers.size(-1):
            #if dimension of codebook and VQ are the same, init the codebook with the pre-computed centers. 
            self.codebook=nn.Parameter(centers,requires_grad=learnable_codebook) #(vocab_size,dim)
            self.dim=centers.size(-1)
            
        elif dim!=centers.size(-1) and learnable_codebook :
            #if learnable codebook and codebook dim different than centers dim, create an embedding table
            self.codebook = nn.Embedding(centers.size(0),dim)
            self.dim=dim
        
        else :
            raise ValueError("Wrong parameters combination for dim and learnable codebook")
        
        self.learnable_codebook=learnable_codebook   
        
        #for compatibility
        self.heads=1
        self.separate_codebook_per_head=False
        
        self.restart = restart
        self.codebook_usage = torch.zeros(self.codebook_size)#register_buffer('codebook_usage',torch.zeros(self.codebook_size))
        self.decay = torch.tensor(0.99) #0.99
        self.beta = torch.tensor(0.25)
        
        #for easier checkpoint loading and compatibility
        self.is_special = is_special #using pre-computed centers specialized for chunk size, dataset and vocab size
        self.data = data #if sepcial, specify the dataset used
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def restart_codebook(self,indexes,new_codevectors):
        with torch.no_grad():
            self.codebook_usage[indexes]=1. #restart usage count
            self.codebook[indexes] = new_codevectors  # Avoids gradient tracking
    
    
    def forward(self, x : torch.Tensor, sample_codebook_temp : float=0.) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        size=x.size()
        
        if len(size)==3 :
            B,L,_=size
            x=x.contiguous().view(B*L,-1) #(B*L,D)
        
        #optimized calculation ||codebook - x||^2 = ||x||^2+||cb||^2 -2*(x*cb)
        dist = - torch.sum(x.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.codebook ** 2, dim=1) + \
                2 * einsum('bd, dn-> bn', x.detach(), rearrange(self.codebook, 'n d-> d n'))
        
        # encoding
        sort_distance, indices = dist.sort(dim=1) #closest embedding to z
        # look up the closest point for the indices
        idx = indices[:,-1]
        encodings = torch.zeros(idx.unsqueeze(1).shape[0], self.codebook_size, device=x.device)
        encodings.scatter_(1, idx.unsqueeze(1), 1) #(B,codebook_size) with ones where a codevector will be assigned
        
        xq = torch.matmul(encodings, self.codebook) #this matmul is equivalent to assigning the closest codevector to the x's
        
        loss = torch.tensor([0.], device = x.device, requires_grad = self.training) #commit loss
        
        if self.training:
            
            commit_quantize = torch.detach(xq) if not self.learnable_codebook else xq #if the codebook has to be optimized by commitment
            
            #STE
            xq = x + (xq - x).detach()
            
            commit_loss = F.mse_loss(commit_quantize,x) 
            #separer le commitment en 2 -> les xq doivent plus se rapprocher des x que l'inverse (surtout quand freeze_bb=False ou condense = attention)
            if self.learnable_codebook:
                commit_loss = self.beta*torch.mean((xq.detach()-x)**2) + torch.mean((xq-x.detach())**2) #beta<1 --> embeddings commit to x more than x to embeddings

            loss = commit_loss

            if self.restart:
                #codebook restart only during training
                
                probs = encodings.mean(dim=0) #avg count of used encodings
                
                self.codebook_usage.to(probs.device).mul_(self.decay.to(probs.device)).add_(probs, alpha=(1-self.decay.to(probs.device))) #moving average update
                
                _,indices = dist.sort(dim=0) #get closest z's to embeddings
                new_features = x.detach()[indices[-1,:]]
                
                #decay based on codebook usage --> less used codebooks update more than those most used
                decay = torch.exp(-(self.codebook_usage*self.codebook_size*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1,self.dim).to(self.codebook.device)
                self.codebook.data = self.codebook.data*(1-decay) + new_features * decay
                
        #reshape to original shape
        if len(size)==3:
            xq = xq.contiguous().view(B,L,-1)
            idx = idx.contiguous().view(B,L)
        
        return xq, idx, loss
    
    def freeze(self):
        self.requires_grad_(False)



#general vector quantizer from lucidrains repo with some modifications for our purposes

#raises problems with whole model (CUDA kernel problem) -> directly modified ludicrains vectorquantize
# class VectorQuantizer(nn.Module):
#     def __init__(self,
#                 dim,
#                 codebook_size,
#                 heads = 1,
#                 separate_codebook_per_head = False,
#                 kmeans_init = False,
#                 kmeans_iters = 10,
#                 sync_kmeans = True,
#                 use_cosine_sim = False,
#                 decay = 0.8,
#                 eps = 1e-5,
#                 threshold_ema_dead_code = 2,
#                 commitment_weight=1.,
#                 diversity_weight=0.1,
#                 reset_cluster_size = None,
#                 use_ddp = False,
#                 learnable_codebook = False,
#                 gumbel_sample = gumbel_sample,
#                 sample_codebook_temp = 1.,
#                 ema_update = True,
#                 affine_param = False,
#                 sync_affine_param = False,
#                 affine_param_batch_decay = 0.99,
#                 affine_param_codebook_decay = 0.9):
        
#         super().__init__()
#         self.dim=dim
#         self.heads=heads
#         self.separate_codebook_per_head=separate_codebook_per_head
#         self.learnable_codebook = learnable_codebook
#         num_codebooks = heads if separate_codebook_per_head else 1
#         self.vocab_size = num_codebooks*codebook_size
#         self.commit_weight=commitment_weight
#         self.diversity_weight = diversity_weight
        
        
#         codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook
        
#         codebook_kwargs = dict(
#             dim = dim,
#             num_codebooks = num_codebooks,
#             codebook_size = codebook_size,
#             kmeans_init = kmeans_init,
#             kmeans_iters = kmeans_iters,
#             sync_kmeans = sync_kmeans,
#             decay = decay,
#             eps = eps,
#             threshold_ema_dead_code = threshold_ema_dead_code,
#             learnable_codebook = learnable_codebook,
#             sample_codebook_temp = sample_codebook_temp,
#             gumbel_sample = gumbel_sample,
#             ema_update = ema_update
#         )
        
        
#         self._codebook = codebook_class(**codebook_kwargs)
        
#     # TODO : HANDLE MULTI HEAD CASE 
#     @property
#     def codebook(self):
#         codebook = self._codebook.embed
#         return codebook
    
    
#     @autocast(enabled = False)
#     def forward(
#         self,
#         x,
#         sample_codebook_temp = None,
#         mask = None,
#         freeze_codebook = False
#     ):
        
        
#         # l2norm for cosine sim, otherwise identity
#         x = self._codebook.transform_input(x)
        
#         quantize, embed_ind, distances = self._codebook(x, sample_codebook_temp, mask, freeze_codebook)
        
#         if self.training:
#             # determine code to use for commitment loss
#             maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

#             commit_quantize = maybe_detach(quantize)            

#             # straight through

#             quantize = x + (quantize - x).detach()
            
#         loss = torch.tensor([0.], device = x.device, requires_grad = self.training)
        
#         if self.training:
#             #compute commit loss
#             commit_loss = F.mse_loss(commit_quantize, x)
            
#             #compute diversity loss
#             #B,L,D = x.size()
#             #soft_dist = torch.softmax(distances.view(B*L,self.codebook.num_codebooks,-1).float(), dim=-1)
#             #perplexity = self.compute_perplexity(soft_dist)
#             diversity_loss = torch.tensor([0.], device = x.device, requires_grad = self.training)#1 - perplexity/self.vocab_size
            
#             loss = self.commit_weight*commit_loss + self.diversity_weight*diversity_loss
        
#         # TODO : handle multi head case 
        
#         return quantize, embed_ind, loss


#from hugging face's w2v2 repo : https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L941
class GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, codevector_dim, num_groups, codebook_size, diversity_weight=0.1):
        super().__init__()
        self.num_groups = num_groups
        self.heads=num_groups #for compatibility with seq2seq architecture
        self.separate_codebook_per_head = True #alwazs true for this vq ?
        self.num_vars = codebook_size
        self.dim = codevector_dim
        self.diversity_weight=diversity_weight

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, codevector_dim // self.num_groups)
        )
        
        self.codebook = self.codevectors[0] #for compatibility
        
        self.weight_proj = nn.Linear(self.dim, self.num_groups * self.num_vars)

        # can be decayed for training
        #self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, sample_codebook_temp=1, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)
        
        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=sample_codebook_temp, hard=True
            ).type_as(hidden_states)
            
            #straight through
            codevector_probs = hidden_states + (codevector_probs - hidden_states).detach()
            codevector_idx = codevector_probs.argmax(dim=-1)
            
            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        #B*L,codebook_size*num_codebooks,dim//num_cb
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors 
        
        #B*L,num_codebooks,codebook_size,dim//num_cb
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        
        #B,L,codebook_dim
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)
        
        #reshape indexes TODO : HANDLE CASE WITH MULTIPLE CPDEBOOKS (ADD CODEBOOK SIZE * 'LAYER', i.e. l0 : idx + 0*vocab_size,l1: idx + 1*vocab_size,...)
        codevector_idx = codevector_idx.view(batch_size, self.num_groups, sequence_length)
        if self.num_groups==1:
            codevector_idx=codevector_idx.squeeze(1)
        
        num_codevectors = self.num_vars * self.num_groups
        diversity_loss = ((num_codevectors - perplexity) / num_codevectors)*self.diversity_weight

        return codevectors, codevector_idx, diversity_loss
