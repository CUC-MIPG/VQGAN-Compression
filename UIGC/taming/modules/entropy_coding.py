import torch
import torchac
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

@torch.no_grad()
def pmf_to_cdf(pmf: torch.Tensor):
    """ convert pmf to cdf, designed for torchac library.

    Args:
        pmf (torch.Tensor): probability, [B, H, W, CodeBook_Size]

    Returns:
        cdf_norm: cdf, [B, H, W, CodeBook_Size + 1]
    """
    if len(pmf.shape) == 4:
        pmf = pmf.unsqueeze(3)
    elif len(pmf.shape) == 5:
        pass
    else:
        raise NotImplementedError("Invalid pmf shape!")
    assert pmf.dtype == torch.float32   # torchac only process float32 data!
    B, H, W, S, L = pmf.shape
    
    cdf = torch.zeros([B, H, W, S, L + 1], device=pmf.device, dtype=torch.float32)
    for i in range(L):
        pmf_slice = pmf[:, :, :, :, :i+1]
        cdf[:, :, :, :, i+1] = torch.sum(pmf_slice, dim=-1)
        
    # normalize, prevent max value > 1.0
    max_cdf = cdf[:, :, :, :, -1].unsqueeze(4)
    cdf_norm = cdf / max_cdf
    
    if S == 1:
       cdf_norm = cdf_norm.squeeze(3) 
    
    return cdf_norm


@torch.no_grad()
def encoding_decoding_torchac(sym:torch.Tensor, pmf:torch.Tensor):
    """ entropy coding with torchac library 
    
    torchac must have all cdf to encode at once, only used in experiment.  
    because in real application, decoder must process elements sequentially,  
    and thus it is impossible to acquire all cdf at once.
    
    Args:
        sym (torch.Tensor): indeces map, [B, H, W]
        pmf (torch.Tensor): probability, [B, H, W, CodeBook_Size]

    Returns:
        byte_stream: compressed bitstream
        byte_num: byte number
    """

    # torchac library can only process symbols in int16 and pmf in float32
    assert sym.dtype == torch.int16
    assert pmf.dtype == torch.float32
    
    cdf = pmf_to_cdf(pmf)
    
    # move data to cpu, torchac can not work on gpu
    sym = sym.cpu()
    cdf = cdf.cpu()
    pmf = pmf.cpu()
    
    # encoding and decoding
    byte_stream = torchac.encode_float_cdf(cdf, sym)
    dec_sym = torchac.decode_float_cdf(cdf, byte_stream)
    byte_num = len(byte_stream)
    assert dec_sym.equal(sym)            # check encode == decode!
    
    # calculate information
    probs_select = torch.gather(pmf, dim=-1, index=sym.unsqueeze(-1).to(dtype=torch.int64)).squeeze(-1)
    information = (-1. * torch.log2(probs_select)).sum().item()
    
    return byte_stream, byte_num, information


@torch.no_grad()
def encoding_masked_torchac(sym:torch.Tensor, pmf:torch.Tensor, mask:torch.Tensor):
    """ entropy coding with torchac library 
    
    torchac must have all cdf to encode at once, only used in experiment.  
    because in real application, decoder must process elements sequentially,  
    and thus it is impossible to acquire all cdf at once.
    
    Args:
        sym (torch.Tensor): indeces map, [B, H, W]
        pmf (torch.Tensor): probability, [B, H, W, CodeBook_Size]
        mask (torch.Tensor): mask, [B, H, W]

    Returns:
        byte_stream: compressed bitstream
        byte_num: byte number
    """

    # torchac library can only process symbols in int16 and pmf in float32
    assert sym.dtype == torch.int16
    assert pmf.dtype == torch.float32
    
    cdf = pmf_to_cdf(pmf)
    
    # move data to cpu, torchac can not working on gpu
    sym = sym.cpu()
    cdf = cdf.cpu()
    pmf = pmf.cpu()
    
    if len(sym.shape) == 3:         # to support VQ [B, H, W]
        sym = sym.unsqueeze(-1)
        cdf = cdf.unsqueeze(-2)
        pmf = pmf.unsqueeze(-2)
        mask = mask.unsqueeze(-1)
    
    sym = sym.flatten(start_dim=0, end_dim=2)
    cdf = cdf.flatten(start_dim=0, end_dim=2)
    pmf = pmf.flatten(start_dim=0, end_dim=2)
    mask = mask.flatten(start_dim=0, end_dim=2)
    unmasked_num = (mask[:, 0]==True).sum().item()
    
    sym_masked = torch.zeros((unmasked_num, sym.shape[-1]), dtype=torch.int16)
    cdf_masked = torch.zeros((unmasked_num, sym.shape[-1], cdf.shape[-1]), dtype=torch.float32)
    pmf_masked = torch.zeros((unmasked_num, sym.shape[-1], pmf.shape[-1]), dtype=torch.float32)
    
    j = 0
    for i in range(len(sym)):
        if mask[i, 0] == True:
            sym_masked[j] = sym[i]
            cdf_masked[j] = cdf[i]
            pmf_masked[j] = pmf[i]
            j += 1
    
    # encoding and decoding
    byte_stream = torchac.encode_float_cdf(cdf_masked, sym_masked)
    dec_sym = torchac.decode_float_cdf(cdf_masked, byte_stream)
    byte_num = len(byte_stream)
    assert dec_sym.equal(sym_masked)            # check encode == decode!
    
    # calculate information
    probs_select = torch.gather(pmf_masked, dim=-1, index=sym_masked.unsqueeze(-1).to(dtype=torch.int64)).squeeze(-1)
    information = (-1. * torch.log2(probs_select)).sum().item()
    
    return byte_stream, byte_num, information


@torch.no_grad()
def pmf_to_quantized_cdf(pmf: torch.Tensor, precision: int = 16) -> torch.Tensor:
    """ convert pmf to quantized cdf, designed for compressai library.
    """
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


@torch.no_grad()
def encoding_decoding_compressai(sym:torch.Tensor, pmf:torch.Tensor):
    """ entropy coding with compressai library
    
    This usually performs worse than torchac, but can encode sequentially
    
    Args:
        sym (torch.Tensor): indeces map, [B, H, W]
        pmf (torch.Tensor): probability, [B, H, W, CodeBook_Size]

    Returns:
        byte_stream: compressed bitstream
        byte_num: byte number
    """
    
    # compressai library can only process symbols in int16 and pmf in float32
    assert sym.dtype == torch.int16
    assert pmf.dtype == torch.float32
    
    encoder = BufferedRansEncoder()
    decoder = RansDecoder()
    sym_list = []
    idx_list = []
    cdf_list = []
    cdf_len_list = []
    offset_list = []
    B, H, W, L = pmf.shape
    
    # move data to cpu
    sym = sym.cpu()
    pmf = pmf.cpu()
    
    # convert tensor to list
    for b in range(B):
        for h in range(H):
            for w in range(W):
                quantized_cdf = pmf_to_quantized_cdf(pmf[b, h, w]).tolist()
                sym_list.append(sym[b, h, w].item())
                idx_list.append(b * (H * W) + h * W + w)
                cdf_list.append(quantized_cdf)
                cdf_len_list.append(len(quantized_cdf))
                offset_list.append(0)
    
    # encoding
    encoder.encode_with_indexes(
        sym_list, idx_list, cdf_list, cdf_len_list, offset_list
    )
    byte_stream = encoder.flush()
    byte_num = len(byte_stream)
    
    # decoding
    decoder.set_stream(byte_stream)
    dec_list = decoder.decode_stream(
        idx_list, cdf_list, cdf_len_list, offset_list
    )
    assert sym_list == dec_list          # check encode == decode!
    
    return byte_stream, byte_num