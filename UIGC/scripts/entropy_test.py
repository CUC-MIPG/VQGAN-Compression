import argparse, os, sys, datetime, glob, importlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import torch
from einops import repeat
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import functional as F

sys.path.append(os.getcwd())
sys.path.insert(0, "../")

from taming.modules.entropy_coding import (
    encoding_decoding_torchac,
    encoding_decoding_compressai
)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# util functions, from sample_fast.py
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_from_function(config, params: dict):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config.params:
        params.update(config.get("params", dict()))
    return globals()[config["target"]](**params)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b", "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t", "--test",
        nargs="*",
        metavar="base_config.yaml",
        help="test config file",
        default=list(),
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        nargs="?",
        help="result output file",
    )
    return parser


# aribirtary bits to bitstream
def to_bitstream(data: np.ndarray, bits_per_sym: int):
    """ write non-integer byte symbol
    """
    data = data.flatten()
    
    # calculate lowest common multiple between byte width (8-bit) and precision
    lcm = np.lcm(8, bits_per_sym)
    syms_per_group = lcm // bits_per_sym    # consider lcm bytes as a group
    group_num = int(np.ceil(len(data) / syms_per_group))
    
    strings_all = bytes()
    
    # process per group
    for g in range(group_num):
        data_slice = data[g:g+syms_per_group] 
        int_slice = 0
        bits_slice = 0
        
        # write symbols in one group
        for i in range(len(data_slice)):
            data_moved = data_slice[i] << bits_per_sym * i
            int_slice += data_moved
            bits_slice += bits_per_sym
        
        bytes_slice = int(np.ceil(bits_slice / 8))
        strings_slice = int(int_slice).to_bytes(bytes_slice, "big")
        strings_all += strings_slice
        
    return strings_all


# ZIP compression
def zip_size(x: torch.Tensor, bits_per_sym: int=10):
    """ compress tensor with zip algorithm.
    """
    # convert tensor to array
    x = x.detach().cpu().numpy().astype('int16')
    
    # define file names
    temp_np_file_name = 'temp_np_save.npy'
    temp_zip_file_name = 'temp_np_save.zip'
    
    # save binary file and use zip to compress it
    # bit_stream = to_bitstream(x, bits_per_sym)    # No write per bit
    # with open(temp_np_file_name, 'bw') as f:
    #     f.write(bit_stream)
    np.save(temp_np_file_name, x)
    os.system(f"zip {temp_zip_file_name} {temp_np_file_name} -9")
    zip_size = os.path.getsize(temp_zip_file_name)  # zip size
    
    # save tensor and compress it using numpy
    np.savez_compressed(temp_np_file_name, x)
    np_compress_size = os.path.getsize(temp_np_file_name + '.npz')   # np compress size
    
    # remove temp files
    os.remove(temp_np_file_name)
    os.remove(temp_zip_file_name)
    os.remove(temp_np_file_name + '.npz')
    
    return zip_size, np_compress_size


# test model
@ torch.no_grad()   # disable auto grad calculation
def test_model(model, dataloader):
    """ test entropy compression performance
    """
    
    device = next(model.parameters()).device
    model.eval()

    col_name = ['img', 'bpp_fix_len', 'bpp_entropy', 'bpp_encode', 'bpp_zip', 'bpp_np']
    result = []

    for i, (img, img_path) in enumerate(dataloader):
        B, _, H, W = img.shape
        pix_num = B * H * W     # pix number of a image
        img_name = img_path[0].split('/')[-1]

        img = img.to(device)
        quant_z, _, info = model.first_stage_model.encode(img)
        z_indices = info[2]
        c_indices = torch.tensor([0], device=device)  # force condition indices to zero

        _, H_indices, W_indices = z_indices.shape   
        index_num = model.first_stage_model.quantize.n_e    # pixel num of index

        pmf_len = model.first_stage_model.quantize.n_e
        pmf_shape = [z_indices.shape[0], z_indices.shape[1], z_indices.shape[2], pmf_len]
        pmf = torch.zeros(pmf_shape, device=device, dtype=torch.float32)

        # loop over all latent index
        for b in range(B):
            for h in range(H_indices):
                for w in range(W_indices):
                    # extract window, 16x16, as paper does
                    window_left = max(0, w - 15)
                    window_right = window_left + 15
                    w_top = max(0, h - 15)
                    w_bottom = w_top + 15

                    # extract pixels like gated pixel CNN
                    # support pixels above current pixel
                    support_indices_vec = z_indices[b, w_top:h, window_left:window_right].reshape(-1)
                    # support pixels in the left of current pixel
                    support_indices_hor = z_indices[b, h, window_left:w].reshape(-1)
                    # flatten pixels
                    support_indices = torch.cat([c_indices, support_indices_vec, support_indices_hor], dim=-1).unsqueeze(0)

                    # calculate probability
                    logits, _ = model.transformer(support_indices)
                    probs = F.softmax(logits[:, -1, :], dim=-1).view(-1)
                    pmf[b, h, w] = probs
                    
        # perform entropy coding
        byte_stream, byte_num, information = encoding_decoding_torchac(
            z_indices.to(dtype=torch.int16),
            pmf.to(dtype=torch.float32)
        )
        enc_bpp = byte_num * 8 / pix_num     # calculate bpp after entropy coding
        entropy_bpp = information / pix_num 
                
        # calculate bpp of fixed length encoding. bit for one symbol is log2(codebook_size)
        fix_bpp = B * H_indices * W_indices * np.log2(index_num) / pix_num

        # calculate zip compressed bpp
        zip_bpp, np_bpp = zip_size(z_indices)
        zip_bpp = zip_bpp * 8 / pix_num
        np_bpp = np_bpp * 8 / pix_num

        result.append([img_name, fix_bpp, entropy_bpp, enc_bpp, zip_bpp, np_bpp])
        if i % 10 == 0:
            print(f"[Testing] {i}/{len(dataloader)}")
        
    result = pd.DataFrame(result, columns=col_name)
    return result


# test masked model
@ torch.no_grad()   # disable auto grad calculation
def test_random_masked_model(model, dataloader, times:int, mask_rate:list):
    """ test entropy compression performance, random mask
    """
    
    device = next(model.parameters()).device
    model.eval()

    col_name = ["image_name"] + [f"{r:.2f}%" for r in mask_rate]
    result = []
    
    def apply_mask(indices, rate, mask_index):
        mask = model.generate_mask(indices, rate)    # generate random mask according to rate
        inv_mask = mask == False
        indices_masked = indices * mask.int() + mask_index * inv_mask.int()
        return indices_masked
    
    def encode_one_image(img, pix_num, rate):
        quant_z, _, info = model.first_stage_model.encode(img)
        z_indices = info[2]
        c_indices = torch.tensor([0], device=device)  # force condition indices to zero
        _, H_indices, W_indices = z_indices.shape   
        index_num = model.first_stage_model.quantize.n_e    # pixel num of index

        pmf_len = model.first_stage_model.quantize.n_e
        pmf_shape = [z_indices.shape[0], z_indices.shape[1], z_indices.shape[2], pmf_len]
        pmf = torch.zeros(pmf_shape, device=device, dtype=torch.float32)
        
        # loop over all latent index
        for b in range(img.shape[0]):   # batch
            for h in range(H_indices):
                for w in range(W_indices):
                    # extract window, 16x16, as paper does
                    window_left = max(0, w - 15)
                    window_right = window_left + 15
                    w_top = max(0, h - 15)
                    w_bottom = w_top + 15

                    # extract pixels like gated pixel CNN
                    # support pixels above current pixel
                    support_indices_vec = z_indices[b, w_top:h, window_left:window_right].reshape(-1)
                    # support pixels in the left of current pixel
                    support_indices_hor = z_indices[b, h, window_left:w].reshape(-1)
                    
                    # flatten pixels
                    support_indices = torch.cat([support_indices_vec, support_indices_hor], dim=-1).unsqueeze(0)
                    
                    # apply mask, only use when items > 5
                    if support_indices.numel() > 5:
                        masked_support_indices = apply_mask(support_indices, rate, mask_index=model.mask_token_idx)
                    else:
                        masked_support_indices = support_indices
                    
                    # flatten pixels
                    support_indices = torch.cat([c_indices.unsqueeze(0), masked_support_indices], dim=-1)

                    # calculate probability
                    logits, _ = model.transformer(support_indices)
                    probs = F.softmax(logits[:, -1, :], dim=-1).view(-1)
                    pmf[b, h, w] = probs
        
        # perform entropy coding
        byte_stream, byte_num = encoding_decoding_torchac(
            z_indices.to(dtype=torch.int16),
            pmf.to(dtype=torch.float32)
        )
        enc_bpp = byte_num * 8 / pix_num     # calculate bpp after entropy coding
        
        # calculate bpp with entropy
        probs_select = torch.gather(pmf, dim=-1, index=z_indices.unsqueeze(-1)).squeeze(-1)
        information = (-1. * torch.log2(probs_select)).sum().item()
        entropy_bpp = information / pix_num 

        # calculate bpp of fixed length encoding. bit for one symbol is log2(codebook_size)
        fix_bpp = B * H_indices * W_indices * np.log2(index_num) / pix_num
        
        return enc_bpp, entropy_bpp, fix_bpp

    print(f"[Testing] random process {times} times.")

    for i, (img, img_path) in enumerate(dataloader):
        img_name = img_path[0].split('/')[-1]
        img = img.to(device)
        
        B, _, H, W = img.shape
        pix_num = B * H * W     # pix number of a image
        
        enc_bpp_list = []
        entropy_bpp_list = []
        
        # loop over all mask rate
        for rate in mask_rate:
            enc_bpp_avg = AverageMeter()        # average for arithmetic encoded bpp
            entropy_bpp_avg = AverageMeter()    # average for probability calculated bpp
            
            # perform random mask several times
            for t in range(times):
                bpp_results = encode_one_image(img, pix_num, rate)
                enc_bpp, entropy_bpp, fix_bpp = bpp_results
                
                enc_bpp_avg.update(enc_bpp)
                entropy_bpp_avg.update(entropy_bpp)
                
                if rate == 0:       # for no masked situation, only encode once.
                    break

            enc_bpp_list.append(enc_bpp_avg.avg)
            entropy_bpp_list.append(entropy_bpp_avg.avg)
            
        result.append([img_name] + enc_bpp_list)
        
        print(f"[Testing] {i}/{len(dataloader)}, finish image {img_name}")
            
    result = pd.DataFrame(result, columns=col_name)
    return result


if __name__ == "__main__":
    """
        Test entropy estimation of cin_transformer.
        Set class label to 0.
    """

    # ----- codes from sample_fast.py, load model. -----
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    configs += [OmegaConf.load(cfg) for cfg in opt.test]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    training_cfg = config.training

    # ----- define VQGAN model -----
    model = instantiate_from_config(config.model).to(config.device).eval()
    model.first_stage_model.quantize.sane_index_shape = True    # important! output 2-d index map

    # ----- define test dataloader -----
    test_data = instantiate_from_config(config.data.test)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, 
        num_workers=1,
        batch_size=1,
        shuffle=False
    )

    # ----- load transformer model -----
    checkpoint = torch.load(training_cfg.best_checkpoint, map_location="cuda")
    model.transformer.load_state_dict(checkpoint['state_dict'])

    # ----- test model -----
    test_params = {
        "model": model,
        "dataloader": test_dataloader
    }
    result = instantiate_from_function(config.test, test_params)
    result.to_excel(config.test.output)
    