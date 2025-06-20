import torch
import torch.nn as nn


class Decision(nn.Module):
    def __init__(self, dim : int, layers : int, vocab_size : int, inner_dim : int = 2048, heads : int = 8, dropout=0.1,decoder_only : bool = False, norm_first : bool = True, relative_pe : bool = True):
        
        super().__init__()
        
        self.dim = dim
        self.layers = layers
        self.inner_dim = inner_dim
        self.heads = heads
        self.decoder_only = decoder_only
        self.norm_first = norm_first
        self.dropout=dropout
        self.relative_pe = relative_pe
        
        if self.decoder_only :
            if self.relative_pe:
                print("Relative position encoding decoder")
                self.decision = RelativePositionTransformerDecoder(layers,dim,heads,inner_dim,dropout,8)
            
            else :
                decoder_layer = nn.TransformerDecoderLayer(dim,nhead=heads, dim_feedforward=inner_dim, batch_first=True,norm_first=norm_first,dropout=dropout)
                self.decision = nn.TransformerDecoder(decoder_layer,layers,norm=nn.LayerNorm(dim))
            
        
        else :
            #warning UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
            self.decision = nn.Transformer(dim,nhead=heads,num_encoder_layers=layers,num_decoder_layers=layers, 
                                           dim_feedforward=inner_dim, batch_first=True,norm_first=norm_first,dropout=dropout)
        
        
        self.output_layer = nn.Linear(self.dim,vocab_size)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def adapt_output_layer(self):
        for p in self.decision.parameters():
            p.requires_grad = False
        
        for p in self.output_layer.parameters():
            p.requires_grad = True
    
    def freeze_last_n_layers(self,n:int):
        total_layers = len(self.decision.layers)
        
        for i, layer in enumerate(self.decision.layers):
            if i < total_layers - n:
                for param in layer.parameters():
                    param.requires_grad = False  # Freeze this layer
            else:
                for param in layer.parameters():
                    param.requires_grad = True  # Keep this layer trainable

    
    def forward(self, src : torch.Tensor, tgt : torch.Tensor, 
                src_mask : torch.Tensor = None, tgt_mask : torch.Tensor = None, 
                src_pad_mask : torch.Tensor = None, tgt_pad_mask : torch.Tensor = None) -> torch.Tensor:
        
        if self.decoder_only:
            memory = src
            out = self.decision(tgt,memory,tgt_mask=tgt_mask,memory_mask=src_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=src_pad_mask)
        
        else :
            out = self.decision(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        
        # add here the output projection ? seems logical : decision "decides" what token to choose  
        out = self.output_layer(out)  
        
        return out

    def encode(self,src,src_mask=None,src_pad_mask=None):
        if not self.decoder_only:
            memory = self.decision.encoder(src,mask=src_mask,src_key_padding_mask=src_pad_mask)
        
        else : memory = src
        
        return memory
    
    def decode(self,tgt,memory,tgt_mask=None,memory_mask=None,tgt_pad_mask=None,memory_pad_mask=None) -> torch.Tensor:
        if not self.decoder_only:
            out = self.decision.decoder(tgt,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_pad_mask)
        else : out = self.decision(tgt,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_pad_mask)
        
        out = self.output_layer(out) #logits
        return out


class RelativePositionTransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, n_heads, inner_dim, dropout, max_relative_position):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativePositionTransformerDecoderLayer(embed_dim, n_heads, inner_dim, dropout, max_relative_position)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(tgt)
    

class RelativePositionTransformerDecoderLayer(nn.Module):
    def __init__(self, hid_dim : int, n_heads : int, inner_dim : int, dropout : float, max_relative_position : int=8):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, max_relative_position)
        self.cross_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, max_relative_position)
        self.ff = nn.Sequential(
            nn.Linear(hid_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hid_dim)
        )
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.norm3 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        
        # Feed-forward
        tgt2 = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        
        return tgt
    
#Relative Positional Encoding MultiHeadAttention
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim : int, n_heads : int, dropout : float, max_relative_positions : int):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_positions

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))#.to(self.device)
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, query, key, value, mask = None, padding_mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale.to(self.device)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        
        #TODO : add padding mask

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x