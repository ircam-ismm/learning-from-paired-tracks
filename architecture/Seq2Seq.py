import torch
import torch.nn as nn
from .Encoder import LocalEncoder
from .Decision import Decision
from typing import Union,Tuple,Optional
import math
import time
from utils.utils import predict_topK_P
from beam_search import BeamSearch, Candidate
from typing import List
from scipy.stats import entropy
import numpy as np

#TODO : IS THIS FUNCTION A METHOD OF SEQ2SEQBASE ?
def create_pad_mask(x:torch.Tensor, eos_idx : int) -> torch.Tensor:
    eos_pos = (x==eos_idx)
    first_eos_pos = torch.argmax(eos_pos.float(),dim=1).unsqueeze(1)
    no_eos = ~eos_pos.any(dim=1) #finds where there is no eos
    
    #create tensor with column indices for optimal pad mask generation
    col_indices = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1).to(x.device)
    
    pad_mask = col_indices > first_eos_pos #there is padding for the positions greater that the eos token
    pad_mask[no_eos]=False #if there is no eos padding mask is False
    pad_mask.to(x.device)
    return pad_mask

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# this is a detrrministic positional embedding, another option could be to use a Embeding layer of shape (max_len,embed_dim) that can be learned
# other more complex solutions exist for relative/local and so positional embeddings : with grouped convolutions (w2v) or relative local attention (Music Transformer)
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.size=max_len

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        x = self.dropout(x)
        return x
    
class EmbeddingPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create an embedding layer for positions
        self.pe = nn.Embedding(max_len, embed_dim)  # (max_len, embed_dim)
        self.size=max_len
        

        # Initialize embeddings (optional, can be left random)
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)  # Like transformer embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pe(positions)  # Add learned positional embeddings
        return self.dropout(x)
    

#TODO : IMPLEMENT generate() METHOD FOR FUTURE USE OF SEQUENCE CONTINUATION PREDICTION
# class for autocompletion task
class Seq2SeqBase(nn.Module):
    def __init__(self, localEncoder : LocalEncoder, decisionModule : Decision, 
                 max_len : int, use_special_tokens : bool = True, has_masking : bool = False): 
        super().__init__()
        
        #TODO : CLEAN THIS. ITS NOT PROPER PYTHON TO DO SUCH
        #information attributes intialized after loading
        self.name = None
        self.segmentation = None
        
        self.encoder = localEncoder
    
        self.dim = self.encoder.dim #keep same dimension as output of encoder
        
        self.pe = PositionalEncoding(self.dim,max_len=max_len)
        
        #transformer bloc for seq2seq modeling
        self.decision=decisionModule 
        
        self.max_len=max_len
        self.codebook_size = localEncoder.quantizer.codebook_size #CAREFUL IF MULTIPLE CODEBOOKS !!!
        
        if localEncoder.quantizer.heads > 1 and localEncoder.quantizer.separate_codebook_per_head == True:
            raise NotImplementedError(f"At the moment this implementation doesnt support multiple codebooks as it probably requires a more complex vocab handling algorithm.")
        
          
        extra_tokens=0
        self.use_special_tokens=use_special_tokens
        if self.use_special_tokens:
            #AUTRE OPTION PROPOSEE PAR TEO : AVOIR UN VECTEUR DE EMBED_DIM AVEC UNE SEULE VALEUR E.G [x,x,....,x] avec x=nn.Parameter et il est appris 
            # avantage c'est que c'est une varaible plus simple a optimiser qu'un enorme vecteur
            self.special_tokens = ["sos", "eos", "pad"]
            self.special_tokens_idx = {}
            self.special_token_embeddings = nn.Embedding(len(self.special_tokens),self.dim)
            for token_idx, attr in enumerate(self.special_tokens):
                self.register_buffer(attr, torch.tensor(token_idx)) #needed in state dict but not trainable. used for embedding retrieval
                self.special_tokens_idx[attr]=torch.tensor(self.codebook_size+extra_tokens) #maybe need to convert to float and send to device here
                extra_tokens+=1
        
        #TODO : COULD BE USEFUL TO CONCATENATE THE VOCAB AND SPECIAL TOKENS TABLE IN ONE
        self.vocab_embedding_table = self.encoder.quantizer.codebook #contains only the vocabulary (not the special tokens)
        
        if has_masking:
            self.spec_mask_embed = nn.Parameter(torch.Tensor(self.dim).uniform_()) #masked indices embedding
        self.has_masking=has_masking
        
        self.vocab_size = self.codebook_size + extra_tokens # vocab from 0 to codebook_size-1 are real tokens and last reamining tokens are special tokens
        
        
    @property
    def device(self):
        return next(self.parameters()).device        
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    
    #tgt=src for this model
    #TODO : it's a bit weird to give as memory (source) the whole input sequence since its the one to be predicted. The model should fijnally learn to predict the memory
    def forward(self, src : torch.Tensor, src_pad_mask : torch.Tensor, 
                sample_codebook_temp : float = None, mask_time_indices : torch.Tensor = None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
         
        #extract quantized vectors
        z_src, src_idx, codebook_loss = self.encoder.forward(src, sample_codebook_temp) #output is a sequence of quantized vectors and corresponding indices in the vocabulary
        
        #here append and prepend sos and eos if applied, along with pad tokens 
        if self.use_special_tokens:
            z_src, src_idx, src_pad_mask = self._apply_special_tokens(z_src,src_idx, src_pad_mask)
            
            
        #add position information
        z_src = self.pe.forward(z_src)
            
        src = z_src
        tgt = src.copy() #tgt=src for autocompletion
        tgt = tgt.detach()
        tgt_pad_mask = src_pad_mask
        
        #the seq2seq transformer predicts every next step so we remove last timestep for it to be predicted (all timesteps are predicted in parallel)
        tgt_input = tgt[:,:-1] 
        tgt_pad_mask = tgt_pad_mask[:,:-1]
        
        src_mask, tgt_mask = self._create_masks(src, tgt_input)
        
        #apply source masking (if there is any)
        if self.has_masking and mask_time_indices.any():
            src[mask_time_indices]=self.spec_mask_embed #(B,S)
            
            T = src.size(1) if not self.decision.decoder_only else tgt_input.size(1)
            src_mask = torch.repeat_interleave(mask_time_indices.unsqueeze(1),repeats=T,dim=1) #(B,T,S)
            #we need to repeat for every head of each example i.e. example 1 -> head1,head2,...,headN, then example 2 --> repeat on batch dimension
            src_mask = torch.repeat_interleave(src_mask,repeats = self.decision.heads,dim=0) #(B*heads,T,S)
        
        out = self.decision(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask) #decision outputs logits over vocab_size
        
        #out = self.output_layer(out) #predicted tokens as logits for each timestep
        
        return out, tgt, src_idx, codebook_loss #return predictions and encoded target sequence for loss computing
    
    def _create_causal_mask(self, sz : int) -> torch.Tensor:
        mask = torch.triu(torch.ones((sz,sz), device=self.device), diagonal=1).to(torch.bool) #equivalent to generate_square_subsequent_mask
        return mask
    
    def _create_masks(self, src : torch.Tensor, tgt : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #src : (B,S,max_samples) and tgt : (B,T, max_samples)
        S,T = src.size(1), tgt.size(1)
        if self.decision.decoder_only:
            src_mask = torch.zeros((T,S), device=self.device).to(torch.bool) #memory mask (T,S)
        
        else : src_mask = torch.zeros((S,S), device=self.device).to(torch.bool) #src self attention can attend to all timesteps
        
        tgt_mask = self._create_causal_mask(T)
        
        return src_mask, tgt_mask   
    
    def _find_first_pad_index(self, pad_mask : torch.Tensor) -> torch.Tensor:
        assert pad_mask.dtype==torch.bool, "Padding mask should be a bool tensor with True where there is padding and False everywhere else"
        #find if there are any true values in the poadding mask (usually at least one sequence has not padding as its the longest sequence)
        
        any_true = torch.any(pad_mask, dim=1) #find if there are true values in the "time" dimension
        
        #find first true occurence (argmax returns the idx of the first occurence of the max value)
        first_true_idx = torch.argmax(1*pad_mask, dim=1)
        
        #handle case where there is no padding with any_true
        first_true_idx = torch.where(any_true, first_true_idx, -1) #return -1 if no padding
        
        return first_true_idx
    
    def _apply_special_tokens(self, z_src : torch.Tensor, src_idx : torch.Tensor, src_pad_mask:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        B = z_src.size(0) #used multiple times
        
        #here we have to use the src and tgt padding mask to correctly assign the eos and pad values
        sos_embed = self.special_token_embeddings(self.sos).expand(B,1,-1) #same sos for all sequneces and same batch size for src and target
        eos_embed = self.special_token_embeddings(self.eos)
        pad_embed = self.special_token_embeddings(self.pad)
        
        sos_idx = self.special_tokens_idx['sos'].expand(B,1).to(self.device)
        eos_idx = self.special_tokens_idx['eos'].to(self.device)
        pad_idx = self.special_tokens_idx['pad'].to(self.device)
        
        #for pad_mask update
        mask_val = torch.tensor(True,device=self.device)
        no_mask_val = torch.tensor(False,device=self.device)
        
        #use the src_pad and tgt_pad mask to find where to put the eos token in each
        #done here or maybe send first pad index from datacollator
        src_first_pad_idx = self._find_first_pad_index(src_pad_mask)
        
        #first append a pad token to z_src and z_tgt for dimension coherence
        #-> rather than handling the -1 index as appending a eos, we append a pad to the whole sequence
        # and then reassign that pad index to eos for the longest sequence(s) (the ones with -1)
        z_src = torch.cat([z_src, pad_embed.expand(B,1,-1)],dim=1)
        
        #do the same thing for src_idx and tgt idx (src/tgt_idx : (B,S/T))
        src_idx = torch.cat([src_idx, pad_idx.expand(B,1)],dim=1)
        
        #update pad mask also
        src_pad_mask = torch.cat([src_pad_mask,mask_val.expand(B,1)],dim=1)
        
        #use pad idx to place eos token and append pad tokens
        for i,src_pad_idx in enumerate(src_first_pad_idx):
            z_src[i,src_pad_idx,:]=eos_embed #replace first pad index with eos (if -1 last step is eos thanks to pad append above)
            src_idx[i,src_pad_idx]=eos_idx #for indexes too      
            src_pad_mask[i,src_pad_idx]=no_mask_val #change padding of eos to False
            if src_pad_idx < z_src.size(1)-1 and src_pad_idx!=-1: #if first pad is not last step and is not lonmgest sequence
                z_src[i,src_pad_idx+1:,:]=pad_embed #next tokens are pad
                src_idx[i,src_pad_idx+1:]=pad_idx #for indexes too
        
        #append sos
        z_src = torch.cat([sos_embed,z_src], dim = 1)
        src_idx = torch.cat([sos_idx,src_idx],dim = 1)
        src_pad_mask = torch.cat([no_mask_val.expand(B,1),src_pad_mask],dim=1)
        
        return z_src, src_idx, src_pad_mask 

    def _from_index_to_embedding(self,index:int) -> torch.Tensor:
        special_tokens_idxs = torch.tensor(list(self.special_tokens_idx.values()),device=self.device)
        
        if index in special_tokens_idxs:
            return self.special_token_embeddings(index - self.codebook_size - 1) #embedding table is of len 3 so we need to shift

        else : return self.vocab_embedding_table[index]
    
    def from_indexes_to_embeddings(self, indexes : torch.Tensor) -> torch.Tensor:
        special_tokens_idxs = torch.tensor(list(self.special_tokens_idx.values()),device=self.device)
        is_special_token = torch.isin(indexes,special_tokens_idxs) #special token positions mask
        
        embeddings = torch.empty(size=indexes.shape+(self.dim,),device=self.device)
        
        embeddings[~is_special_token] = self.vocab_embedding_table[indexes[~is_special_token]] #insert vocab embedding if idx in vocab range
        embeddings[is_special_token] = self.special_token_embeddings(indexes[is_special_token] - self.codebook_size) #insert spec token embed if idx in spec tokens idxs
        
        return embeddings
    
    def _greedy_decoding(self, memory : torch.Tensor, memory_pad_mask : torch.Tensor, k:Union[int,float], max_len : Optional[int],
                         gt_set : torch.Tensor = None, temperature : float = 1.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
            
        B = memory.size(0)
        
        if gt_set != None:
            if gt_set.ndim < 2 : #1D tensor
                gt_set = gt_set.view(1,-1).repeat(B,1) # (B, set size)                
        
        #init tgt as SOS
        tgt = self.special_token_embeddings.forward(self.sos).unsqueeze(0).expand(B,1,-1) #(B,1,D)
        tgt_idx = torch.full((B,1),fill_value=self.special_tokens_idx["sos"],device=self.device) #(B,1)
        
        #initialize first logits as zeros for SOS
        probs = torch.zeros(B,1,self.vocab_size,device=self.device)
        probs[:,:,self.special_tokens_idx["sos"]]=1 #assign 1 to SOS
        
        tgt_pad_mask = None #init pad mask as none
        #next_token = torch.empty((B,self.dim), dtype=tgt.dtype, device=memory.device) #init next token tensor
        finished = torch.zeros(B, dtype=torch.bool, device=self.device) #mask for finished sequences
        eos_idx = self.special_tokens_idx["eos"]
        pad_idx = self.special_tokens_idx["pad"]
        #special_tokens_idxs = torch.tensor(list(self.special_tokens_idx.values()),device=self.device) #(3,)
        if max_len==None : max_len=self.max_len
        
        while tgt.size(1)<max_len:
            
            tgt_pe = self.pe(tgt) #apply pos encoding (B,T,dim)
            
            #create tgt mask
            tgt_mask = self._create_causal_mask(tgt_pe.size(1))
            
            #predict logits
            logits = self.decision.decode(tgt_pe,memory,tgt_mask,
                                          tgt_pad_mask=tgt_pad_mask,
                                          memory_pad_mask=memory_pad_mask)[:,-1:,:] #(B,1,vocab_size) only take last step
                        
            #apply temperature
            logits = logits/temperature
            
            #append new probs
            probs = torch.cat([probs,logits.softmax(-1)],dim=1)
            
            #tgt_token = tgt_gt[:,tgt.size(1)].unsqueeze(1) if tgt_gt != None else None #(B,1) #old force_coupling
            #top-K prediction
            next_token_idx = predict_topK_P(k,logits, gt_set, from_set = gt_set != None).reshape(logits.shape[:-1])[:,-1]  #(B,)

            #next_token : (B,D)
            next_token = self.from_indexes_to_embeddings(next_token_idx)
            
            #replace finished sequences next token by a pad token/idx
            next_token_idx[finished] = pad_idx
            next_token[finished]=self.special_token_embeddings(self.pad)
            
            #append next_token to tgt
            tgt = torch.cat([tgt,next_token.unsqueeze(1)],dim=1) #(B,T+1,D)
            tgt_idx = torch.cat([tgt_idx,next_token_idx.unsqueeze(1)],dim=1) #(B,T+1)
            
            #update padding_mask (B,T)
            tgt_pad_mask = tgt_idx == pad_idx #padding where tgt_idx is pad
            
            #update finished at end
            finished = finished | (next_token_idx==eos_idx)
            
            #break the loop if all sequences came to an end
            if finished.all():
                break
        
        return tgt, tgt_idx, probs
    
    def apply_repetition_penalty(self, probs: torch.Tensor, states: torch.Tensor, penalty: float) -> torch.Tensor:
        """
        Reduces probability of already generated tokens to discourage repetition.
        
        Args:
            probs (torch.Tensor): Shape (B, beam_width, vocab_size), probabilities from softmax.
            states (torch.Tensor): Shape (B * beam_width, T), previously generated sequences.
            penalty (float): Repetition penalty factor (e.g., 1.2 means reducing repeated tokens).
        
        Returns:
            torch.Tensor: Adjusted probabilities with repetition penalty.
        """
        B, beam_width, vocab_size = probs.shape

        # Convert to shape (B * beam_width, vocab_size)
        probs = probs.view(B * beam_width, vocab_size)

        for i in range(B * beam_width):
            for token in set(states[i].tolist()):  # Unique tokens in the sequence
                #print(f"before penalty on token {token}",probs[i, token])
                probs[i, token] /= penalty  # Reduce probability of repeated tokens
                #print(f"after penalty on token {token}",probs[i, token])

        return probs.view(B, beam_width, vocab_size)  # Restore shape
    
    # beam search transition function : computes the probabilities over the state space given an input sequence of states 
    def __beam_search_transition_fn(self, 
                                    candidates: List[List[Candidate]], 
                                    memory : torch.Tensor, 
                                    memory_mask : torch.Tensor,
                                    temperature : float,
                                    repetition_penalty : float = 1.5) -> torch.Tensor:
        
        B,T_src,dim=memory.shape
        beam_width = len(candidates[0])
        
        
        #get sequence of states from each candidate
        states = torch.tensor(
            [[candidate.states for candidate in candidates_batch] for candidates_batch in candidates],
                              device=self.device
                              ) #(B, beam_width,T)
        #print(states)
        #expand memory to match states shape
        memory = memory.unsqueeze(1).repeat(1,beam_width,1,1) #(B, beam_width, T_src,dim)
        memory_mask = memory_mask.unsqueeze(1).repeat(1,beam_width,1) #(B,beam_width,T_src)
        
        #reshape
        states = states.contiguous().view(B*beam_width,-1) #(B*beam_width,T)
        memory = memory.contiguous().view(B*beam_width,T_src,dim) #(B*beam_width,T_src,dim)
        memory_mask = memory_mask.contiguous().view(B*beam_width,T_src) #(B*beam_width,T_src)
        
        #convert states to vectors
        tgt = self.from_indexes_to_embeddings(states) #(B*beam_width,T,dim)
        tgt_pe = self.pe(tgt) #apply positional encoding
        
        #generate tgt masks
        tgt_mask = self._create_causal_mask(tgt_pe.size(1))
        tgt_pad_mask = create_pad_mask(states, self.special_tokens_idx['eos'].item())
        
        #we dont need to update directlz the states after eos it can be done outside the transition_fn.
        #what matters is the generation of the padding mask to correctly generate representations with attention
        
        logits = self.decision.decode(tgt_pe, 
                                      memory, 
                                      tgt_mask, 
                                      tgt_pad_mask=tgt_pad_mask,
                                      memory_pad_mask=memory_mask) # (B*beam_width,T,vocab_size) 
        
        logits = logits.view(B,beam_width,logits.size(1), logits.size(2)) #(B,beam_width,T,vocab_size)
        logits = logits/temperature
        #only take last step for each elem in batch and beam
        probs = torch.softmax(logits,dim=-1)[:,:,-1,:] #(B,beam_width,vocab_size)
        
        #apply repetition penalty
        probs = self.apply_repetition_penalty(probs, states, repetition_penalty)
        
        #print(probs.shape)
        #print(probs)
        #print(torch.topk(probs,k=5))
        
        return probs
    
    #custom score function for beam search decoding
    def __beam_search_custom_fn(self,candidate:Candidate,entropy_weight:float=0.):
        probs = torch.tensor(candidate.compute_prob())
        llh=torch.log(probs)/(candidate.effective_length**0.75)
        
        states = candidate.states
        states_count = torch.bincount(torch.tensor(states[:candidate.effective_length]))
        H = entropy(states_count/sum(states_count))/torch.log(torch.tensor(len(states_count)+1e-9))
        
        return llh + entropy_weight*H

    
    def _beam_search_decoding(self, memory : torch.Tensor, memory_pad_mask : torch.Tensor, 
                              k : int, max_len : int, temperature : float, entropy_weight : float = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B = memory.size(0)
        
        #init with SOS
        x_init = torch.tensor([[self.special_tokens_idx['sos'].item()] for _ in range(B)], device=self.device)
        eos = self.special_tokens_idx['eos'].item()
        
        #kwargs
        trans_fn_args = {'memory':memory, 'memory_mask':memory_pad_mask,"temperature":temperature}
        score_fn_args = {'entropy_weight':entropy_weight} 
        
        #init beam search
        beamsearch = BeamSearch(self.__beam_search_transition_fn, trans_fn_args, terminal_state = eos, score_fn=self.__beam_search_custom_fn,score_fn_args=score_fn_args)

        #find best candidate sequences
        best_candidates = beamsearch(x_init, k, max_len) #(B,nbest) with nbest = 1
        
        tgt_idx = torch.tensor([[c.states for c in nbest_candidate] for nbest_candidate in best_candidates],device=self.device).squeeze(1) #remove extra dimension
        tgt = self.from_indexes_to_embeddings(tgt_idx)
        
        probs = torch.tensor(np.array([[c.beam_probs for c in nbest_candidate] for nbest_candidate in best_candidates]),device=self.device).squeeze(1)
        
        return tgt, tgt_idx, probs

    
    def decode(self, memory : torch.Tensor, memory_pad_mask : torch.Tensor,
               k:int, max_len : int, decoding_type : str, temperature : float = 1,
               gt_set : torch.Tensor = None, entropy_weight : Optional[float] = 0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if decoding_type == "greedy":
            tgt_out, tgt_idx, probs = self._greedy_decoding(memory, memory_pad_mask, k, max_len, gt_set, temperature)
            
        elif decoding_type == "beam":
            tgt_out, tgt_idx, probs = self._beam_search_decoding(memory, memory_pad_mask, k, max_len, temperature, entropy_weight)
        
        else : raise ValueError(f"Wrong 'decoding_type' argument {decoding_type}. Should be 'greedy' or 'beam'")
        
        return tgt_out, tgt_idx, probs
    
    #encode input sequence --> assign labels (codebook index,..)
    @torch.no_grad
    def encode(self, src : torch.Tensor, src_pad_masks : List[torch.Tensor], both :bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: #needs both pad masks
        z_src, src_idx, codebook_loss = self.encoder.forward(src, padding_mask = src_pad_masks[0])        
        
        if self.use_special_tokens or both:
            z_src_, src_idx_, src_pad_mask_ = self._apply_special_tokens(z_src,src_idx, src_pad_masks[1])
        
        else : 
            #if not apply special tokens then src_pad_mask stays the same
            src_pad_mask_ = src_pad_masks[1]
            z_src_ = z_src
            src_idx_ = src_idx
        
        if both :
            return (z_src, z_src_), (src_idx, src_idx_), (src_pad_mask_,src_pad_mask_)
                                                                 
        else :
            return z_src_, src_idx_, src_pad_mask_
    
    def scheduled_sampling_forward(self, src : torch.Tensor, tgt : torch.Tensor, 
                src_mask : torch.Tensor, tgt_mask : torch.Tensor, src_pad_mask : torch.Tensor, tgt_pad_mask : torch.Tensor,
                scheduled_prob : float
                ):
        """
        tgt: (B, T) -> Ground truth target tokens
        memory: (B, T_src, dim) -> Encoder output
        memory_mask: (B, T_src) -> Mask for encoder memory
        scheduled_prob: float -> Probability of using model-generated token instead of ground truth
        """
        B, T, dim = tgt.shape

        with torch.no_grad():
            # Generate logits for the entire sequence (SINGLE FORWARD PASS)
            logits = self.decision.forward(src,tgt,src_mask,tgt_mask,src_pad_mask, tgt_pad_mask)  # (B, T, vocab_size)
            # Get model predictions (argmax over vocabulary)
            pred_tokens = logits.argmax(dim=-1)  # (B, T)
        
        sampled_tgt = tgt.clone()  # Start with ground truth

        # Create a mask deciding where to use model predictions
        mask = (torch.rand(B, T, device=tgt.device) < scheduled_prob)  # True = use model output

        # Avoid modifying <SOS> token (assume first token is always ground truth)
        mask[:, 0] = False  # Ensure first token remains teacher-forced
        
        #print("schedeled sampling mask:",mask)
        
        #Shift pred_tokens to be correctly aligned with sampled_tgt
        pred_tokens = torch.cat([torch.full((B,1),self.special_tokens_idx['sos'],device=tgt.device), pred_tokens[:, :-1]], dim=1)

        # Replace tokens with model predictions where mask is True
        sampled_tgt[mask] = self.from_indexes_to_embeddings(pred_tokens[mask])
        

        return sampled_tgt
    
    
    
class Seq2SeqCoupling(Seq2SeqBase):
    def __init__(self, localEncoder:LocalEncoder, decisionModule : Decision, 
                 max_len : int, use_special_tokens : bool = True, has_masking : bool = False): #add params for transformer
        
        super().__init__(localEncoder, decisionModule, max_len, use_special_tokens, has_masking)
       
    def _apply_masking(self, x : torch.Tensor, mask_time_indices : torch.Tensor, tgt_input : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.decision.decoder_only : assert tgt_input != None, "if decoder only, tgt_input should be specified"
        #take sos and eos into account
        mask_time_indices = torch.cat([torch.tensor([False],device=self.device).expand(x.size(0),1),
                                        mask_time_indices,
                                        torch.tensor([False],device=self.device).expand(x.size(0),1)],dim=1)
        
        x[mask_time_indices]=self.spec_mask_embed
        
        T = x.size(1) if not self.decision.decoder_only else tgt_input.size(1) #self attention if decoder only and cross atention for decodr only
        src_mask = torch.repeat_interleave(mask_time_indices.unsqueeze(1),repeats=T,dim=1) #(B,T,S)
        #we need to repeat for every head of each example i.e. example 1 -> head1,head2,...,headN, then example 2 --> repeat on batch dimension
        src_mask = torch.repeat_interleave(src_mask,repeats = self.decision.heads,dim=0) #(B*heads,T,S)
        
        return x, src_mask
    
    def forward(self, src : torch.Tensor, tgt : torch.TensorType, 
                src_pad_masks : List[torch.Tensor], tgt_pad_masks : List[torch.Tensor], 
                sample_codebook_temp : Optional[float] = None, 
                mask_time_indices : Optional[torch.Tensor] = None,
                scheduled_sampling_prob : Optional[float] = None ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
         
        #sample dim padding masks
        src_pad_mask=src_pad_masks[0]
        tgt_pad_mask=tgt_pad_masks[0]
        
        #extract quantized vectors
        z_src, src_idx, codebook_loss = self.encoder(src, sample_codebook_temp, src_pad_mask) #output is a sequence of quantized vectors and corresponding indices in the vocabulary
        z_tgt, tgt_idx, _ = self.encoder(tgt, sample_codebook_temp, tgt_pad_mask) #tgt index will be used for crossentropy loss
        
        #chunk dim padding masks
        src_pad_mask = src_pad_masks[1]
        tgt_pad_mask=tgt_pad_masks[1]
        
        #here append and prepend sos and eos if applied, along with pad tokens 
        if self.use_special_tokens:
            z_src, src_idx, src_pad_mask = self._apply_special_tokens(z_src,src_idx, src_pad_mask)
            z_tgt, tgt_idx, tgt_pad_mask = self._apply_special_tokens(z_tgt,tgt_idx, tgt_pad_mask)
            
        
        #add position information (if not trainable relative pos encoding)
        if not self.decision.relative_pe:
            z_src = self.pe.forward(z_src)
            z_tgt = self.pe.forward(z_tgt)
        
        #detach targets -> avoid gradient flowing from answers
        z_tgt = z_tgt.detach()
        tgt_idx = tgt_idx.detach()
        
        #rename to src and tgt
        src = z_src
        tgt = z_tgt
        
        #the seq2seq transformer predicts every next step so we remove last timestep for it to be predicted (all timesteps are predicted sequentially)
        tgt_input = tgt[:,:-1] 
        tgt_pad_mask = tgt_pad_mask[:,:-1]
                
        src_mask, tgt_mask = self._create_masks(src, tgt_input)
        
        #apply scheduled sampling here
        if scheduled_sampling_prob != None:
            tgt_input = self.scheduled_sampling_forward(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, scheduled_sampling_prob)
        
        
        #apply source masking
        if self.has_masking:
            src, src_mask = self._apply_masking(src, mask_time_indices, tgt_input)
            
            
        #compute probs over vocab space
        out = self.decision.forward(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask) 
        
        return out, tgt, tgt_idx, codebook_loss #return predictions and encoded target sequence for loss computing
    
    
    #generate sequence of labels to "couple" the input sequence of labels (memory)
    @torch.no_grad
    def coupling(self, encoded_src : torch.Tensor, src_pad_mask : torch.Tensor, #and this pad mask is on chunks dim (after process from encode)
                 k:int, max_len : int, decoding_type : str, temperature : float, 
                 gt_set : Optional[torch.Tensor] = None, entropy_weight : Optional[float] = 0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        
        src = encoded_src
        
        #apply position to src
        src = self.pe.forward(src)
        
        memory = self.decision.encode(src,src_mask=None,src_pad_mask=src_pad_mask) #encode src once -> pass through Transformer encoder if enc-dec else will be = src
        
        tgt, tgt_idx, probs = self.decode(memory, src_pad_mask, k, max_len, decoding_type, temperature, gt_set, entropy_weight)
        
        return tgt, tgt_idx, probs
    
    #encode + coupling
    @torch.no_grad 
    def generate(self,src : torch.Tensor ,src_pad_masks : List[torch.Tensor], 
                 k : int, decoding_type : str, max_len : Optional[int] = None, 
                 temperature : float = 1, entropy_weight : float = 0, gt_set : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,torch.Tensor, torch.Tensor]:
        
        encoded_src, src_idx, src_pad_mask = self.encode(src, src_pad_masks = src_pad_masks)  #encode audio sequence into sequence of labels / codevectors (and process chunks pad mask)
        
        if not max_len : max_len = encoded_src.size(1) #maximum generated sequence size is equal to size of input sequence
        
        tgt, tgt_idx, probs = self.coupling(encoded_src, src_pad_mask, k, max_len, decoding_type, temperature, gt_set, entropy_weight) #generate sequence of expected labels for coupling
        
        return tgt, tgt_idx, probs #tgt probably not used but not bad idea    
        
        
        

