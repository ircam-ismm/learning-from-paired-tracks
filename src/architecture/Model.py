from .Encoder import LocalEncoder,Backbone
from .Decision import Decision
from .VectorQuantizer import KmeansQuantizer # VectorQuantizer, GumbelVectorQuantizer, 
from .Seq2Seq import Seq2SeqBase, Seq2SeqCoupling 
import numpy as np
from fairseq.checkpoint_utils import load_model_ensemble_and_task
#from vector_quantize_pytorch import VectorQuantize # TODO : ADD OTHER OPTIONS OF VQ
import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, Union
from pathlib import Path

#wrapper class to get eattributes without changing whole code
class myDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

#script to build different models and load checkpopints

#TODO : NOW THE SEQ2SEQ BUILDER CAN TAKE **KWARGS FROM DICT
def load_model_checkpoint(ckp_path:Path, backbone_checkpoint="/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt",vq_ckp:str=None) -> Tuple[Union[Seq2SeqBase,Seq2SeqCoupling], dict, dict] :
    """
    Function to load pre-trained checkpoint of Seq2Seq/Seq2SeqCoupling

    Args:
        ckp_path (Path): Path to checkpoint
        backbone_checkpoint (str, optional): The pretrained encoder checkpoint. Defaults to "/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt" for the Wav2Vec 2.0 pre-trained on music.

    Returns:
        Tuple[Union[Seq2SeqBase,Seq2SeqCoupling], dict, dict]: Returns the pretrained module, its parameters dict and optimizers dict
    """
    
    ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
    model_class = ckp["model_class"]
    state_dict = ckp["state_dict"]
    model_params = ckp["model_params"]
    optimizer_state_dict = ckp['optimizer']
    
    #bb_ckp="../w2v_music_checkpoint.pt"

    bb_type=model_params["backbone_type"]
    max_len = model_params["max_len"]
    dim=model_params["dim"]
    if dim == 0 : dim = 768
    vocab_size=model_params["vocab_size"]
    encoder_head=model_params["encoder_head"]
    condense_type = model_params["condense_type"]
    use_special_tokens=model_params["use_special_tokens"]
    has_masking=model_params['has_masking']
    decoder_only=model_params["decoder_only"]
    transformer_layers=model_params["transformer_layers"]
    inner_dim=model_params["inner_dim"]
    heads = model_params["heads"]
    try:
        dropout = model_params["dropout"]
    except :
        dropout = 0.1 #default value for older models
    
    try :
        chunking = model_params['pre_post_chunking']
    except :
        chunking = 'pre' #old models have pre chunking
        
    try :
        special_vq = model_params['special_vq']
        data = model_params["vq_data"]
        assert data is not None, "If special VQ, specify which data is the VQ from..."
    except :
        special_vq = False
        data = None
    
    try :
        relative_pe = model_params["relative_pe"]
    except:
        relative_pe = False
    
    task = model_params["task"]
    
    #if issubclass(model_class,Seq2SeqBase):
    model = SimpleSeq2SeqModel(backbone_checkpoint,bb_type,dim,vocab_size,max_len,encoder_head,use_special_tokens,chunking=chunking,
                                   condense_type=condense_type,has_masking=has_masking,task=task,
                                   transformer_layers=transformer_layers,decoder_only=decoder_only, relative_pe=relative_pe,inner_dim=inner_dim,heads=heads,dropout=dropout,
                                   special_vq=special_vq,chunk_size = model_params["chunk_size"], data = data,VQpath=vq_ckp)
    
    #else : raise ValueError(f"the model class from the checkpoint is invalid. Should be an instance (or subclass) of 'Seq2SeqBase' but got {model_class}")
    
    model.load_state_dict(state_dict)
    
    segmentation_startegy = model_params["segmentation"]
    
    #information attributes
    model.segmentation = segmentation_startegy
    model.name = model_params["run_id"]
    
    return model, model_params, optimizer_state_dict

def build_backbone(checkpoint : Path, type : str, mean : bool, pooling : bool, output_final_proj : bool, fw : str = "fairseq") -> Backbone:
    """
    Function to build Bakcbone module

    Args:
        checkpoint (Path): pre-trained backbone checkpoint.
        type (str): backbone type. keyword for backbone, e.g. w2v for Wav2Vec 2.0 checkpoint.
        mean (bool): True to apply mean accross time dimension of the output.
        pooling (bool): True to apply max pooling accross the time dimension of the output.
        output_final_proj (bool): True to apply final projection layer from backbone (usually for wav2vec).
        fw (str, optional): framework keyword. Defaults to "fairseq".

    Returns:
        Backbone: Bakcbone module
    """
    #load pretrained backbone
    if fw=="fairseq":
        models, _, _ = load_model_ensemble_and_task([checkpoint])
        pretrained_backbone = models[0]
    
    else :
        NotImplementedError("Not implemented builder for other framework than fairseq")
    
    backbone = Backbone(pretrained_backbone,type,mean,pooling,output_final_proj)
    
    return backbone

def build_quantizer(dim : int, vocab_size : int, learnable_codebook : bool, restart : bool, 
                    is_special: bool, chunk_size : float = None, data : str = None, path : str = None)-> KmeansQuantizer:
    """
    Function to build the Vector Quantizer module from the pre-computed centers from kmeans.

    Args:
        dim (int): dimension of the centers
        vocab_size (int): size of the vocab. Should be among [16,32,64,128,256,512,1024]
        learnable_codebook (bool): If the codebook should be optimized.
        restart (bool): If the codebook entries should be restarted (see "Jukebox")
        is_special (bool): If the pre-computed centers are from the specialized training, i.e. centers optimized per segmentation size and dataset.
    chunk_size (float, optional): size of the segmentation window for the special VQ. Defaults to None.
        data (str, optional): The dataset keyword : moises or canonne, for special VQ. Defaults to None.
        path (str, optionale): path to the trained VQ if other than moises or canonne.

    Returns:
        KmeansQuantizer: VQ from pre-computed centers.
    """
    #vector quantizer  
    assert vocab_size in [16,32,64,128,256,512,1024]
    if is_special:
        assert chunk_size in [0.1,0.25,0.35,0.5]
        assert data in ["canonne", "moises"]
        print("Special VQ")
        centers=np.load(f"clustering/kmeans_centers_{vocab_size}_{chunk_size}s_{data}.npy",allow_pickle=True)
    elif path == None :
        centers=np.load(f"clustering/kmeans_centers_{vocab_size}_{dim}.npy",allow_pickle=True)
    else :
        centers = np.load(path, allow_pickle = True)
        
    centers=torch.from_numpy(centers)
    vq = KmeansQuantizer(centers,learnable_codebook,dim,restart,is_special, data)
    
    return vq
    

def build_localEncoder(backbone_ckp : Path, backbone_type : str, freeze_backbone : bool, dim : int, 
                       vocab_size : int, learnable_codebook : bool, restart_codebook : bool,
                       chunking : str = "post", encoder_head : str = "mean", condense_type : str = None, 
                       special_vq: bool = True, chunk_size : float = None, data : str = None, path : str = None) -> LocalEncoder:
    """
    Function to build the localEncoder module (equivalent to the Perception module from "Learning Relationships Between Separate AudioTracks for Creative Applications" (Bujard, 2025))

    Args:
        backbone_ckp (Path): checkpoint to the pre-trained backbone
        backbone_type (str): backbone type. keyword for backbone, e.g. w2v for Wav2Vec 2.0 checkpoint.
        freeze_backbone (bool): True to freeze backbone
        dim (int): dimension of the module. 256 or 768
        vocab_size (int): size of the codebook from the VQ
        learnable_codebook (bool): True to optimize codebook
        restart_codebook (bool): True to restart dead codebook entries.
        chunking (str, optional): which kind of chunking. choose between pre and post chunking. Defaults to "post".
        encoder_head (str, optional): temporal information condenser. Defaults to "mean".
        condense_type (str, optional): if attention condenser, which type : weighed or mask. Defaults to None.
        special_vq (bool, optional): if speical vq is used. Defaults to True.
        chunk_size (float, optional): segmentation size. Defaults to None.
        data (str, optional): dataset used for centers kmeans. Defaults to None.
        path (str, optionale): path to the trained VQ if other than moises or canonne.

    Raises:
        ValueError: if not freeze_backbone and not learnable_codebook

    Returns:
        LocalEncoder: localEncoder (Perception) module
    """
    
    output_final_proj = dim==256 #if model dimension is 256 we want the final projection output, else 768 hidden layer output dim
    
    #load pretrained backbone
    backbone=build_backbone(backbone_ckp,backbone_type,
                            mean=False,pooling=False, 
                            output_final_proj=output_final_proj,
                            fw="fairseq") #no mean or pooling for backbone in seq2seq, collapse done in encoder
    
    if freeze_backbone:
        backbone.eval() # SI ON UNFREEZE BB IL FAUT TRAIN VQ
        backbone.freeze() #freeze backbone
    
    elif learnable_codebook == False:
        raise ValueError("Train VQ if backbone in learning.")
    
    else : #trainable bb and codebook -> only freeze feature extractor (CNN)
        backbone.freeze_feature_extractor()
       
    
    #vector quantizer  
    vq = build_quantizer(dim, vocab_size, learnable_codebook,restart_codebook, special_vq, chunk_size, data, path)
    
    localEncoder=LocalEncoder(backbone,vq,encoder_head,embed_dim=backbone.dim,condense_type=condense_type,chunking_pre_post_encoding=chunking)
    
    return localEncoder


#create class for decision module to handle forward call in seq2seq
def build_decision(dim : int, layers : int, vocab_size : int, 
                   inner_dim : int = 2048, heads : int = 8, dropout : float = 0.1, decoder_only : bool = False, 
                   norm_first : bool = True, relative_pe : bool = False) -> Decision:
    
    decisionModule = Decision(dim, layers, vocab_size, inner_dim, heads, dropout, decoder_only, norm_first, relative_pe)
    return decisionModule
    


def SimpleSeq2SeqModel(backbone_checkpoint : Path,
                       backbone_type : str, 
                       dim : int,
                       vocab_size : int,
                       max_len : int,
                       encoder_head : str,
                       use_special_tokens : bool,
                       task : str,
                       chunking : str,
                       restart_codebook : bool = False,
                       condense_type : str = None,
                       has_masking : bool = False,
                       freeze_backbone : bool = True,
                       learnable_codebook : bool = False,
                       transformer_layers : int = 6,
                       dropout : float = 0.1,
                       decoder_only : bool = True,
                       inner_dim : int = 2048,
                       heads : int = 12,
                       norm_first : bool = True,
                       special_vq : bool = True,
                       chunk_size : float = None,
                       data : str = None,
                       VQpath : str = None,
                       relative_pe : bool = False,
                    #    kmeans_init : bool = False,
                    #    threshold_ema_dead_code : float = 0,
                    #    commit_weight : float = 1.,
                    #    diversity_weight :  float = 0.1
                    ) -> Union[Seq2SeqBase,Seq2SeqCoupling]: 
    """
    Function to build Seq2Seq model

    Args:
        backbone_checkpoint (Path): backbone checkpoint pre-trained on music
        backbone_type (str): backbone type.
        dim (int): model dimension, either 256 or 768
        vocab_size (int): vocabulary size
        max_len (int): maximum output length
        encoder_head (str): information condenser head. attention, mean or pooling
        use_special_tokens (bool): True to use special tokens sos, eos and pad
        task (str): completion or coupling task
        chunking (str): pre or post chunking of the encoded sequences
        restart_codebook (bool, optional): restart dead codebook entries. Defaults to False.
        condense_type (str, optional): if encoder_head=="attention", specify which type : "weighed" or "mask". Defaults to None.
        has_masking (bool, optional): mask some timesteps of the source vectors. Defaults to False.
        freeze_backbone (bool, optional):. Defaults to True.
        learnable_codebook (bool, optional): True to optimize codebook. Defaults to False.
        transformer_layers (int, optional): number of transformer layers. Defaults to 6.
        dropout (float, optional): dropout percentage. Defaults to 0.1.
        decoder_only (bool, optional): if the transfoprmer has only a decoder. Defaults to True.
        inner_dim (int, optional): feedforward dimension of transformer. Defaults to 2048.
        heads (int, optional): number of heads of the transformer. Defaults to 12.
        norm_first (bool, optional): apply pre-layer normalization. Defaults to True.
        special_vq (bool, optional): specialized vq. Defaults to True.
        chunk_size (float, optional): segmentation window size. Defaults to None.
        data (str, optional): datset for specialized vq. Defaults to None.
        VQpath (str, optionale): path to the trained VQ if other than moises or canonne
        relative_pe (bool, optional): apply relative positional encoding. Defaults to False.
       
    Returns:
        Union[Seq2SeqBase,Seq2SeqCoupling]: Sequence modeling module for autompletion or coupling.
    """
    
    
    assert task.lower() in ["coupling","completion"]
    assert chunking in ['pre','post']
    
    localEncoder=build_localEncoder(backbone_checkpoint,backbone_type, freeze_backbone, dim,
                                    vocab_size, learnable_codebook, restart_codebook, chunking,
                                    encoder_head, condense_type, 
                                    special_vq, chunk_size, data, VQpath)
        
    decision_module = build_decision(localEncoder.dim,transformer_layers,
                                     vocab_size=vocab_size+3*use_special_tokens, #+ pad, sos, eos
                                     inner_dim=inner_dim,
                                     heads=heads,
                                     dropout=dropout,
                                     decoder_only=decoder_only,
                                     norm_first=norm_first,
                                     relative_pe=relative_pe)
    
    model_class = Seq2SeqCoupling if task == "coupling" else Seq2SeqBase
    
    seq2seq = model_class(localEncoder, decision_module, max_len, use_special_tokens=use_special_tokens,has_masking=has_masking)
    return seq2seq
