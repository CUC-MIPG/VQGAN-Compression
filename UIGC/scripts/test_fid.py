import argparse, os, sys, datetime, glob, importlib
import pathlib
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_activation_statistics,
    calculate_frechet_distance
)

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--src",
        type=str,
        const=True,
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--tar",
        type=str,
        const=True,
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--pfx",
        type=str,
        const=True,
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--device",
        type=str,
        const=True,
        default="",
        nargs="?",
    )
    return parser


def get_img_list(path, pfx=None):
    img_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            if file_ext.replace(".", "") in IMAGE_EXTENSIONS:
                if pfx:
                    if file_name.endswith(pfx):
                        img_list.append(file_path)
                else:
                    img_list.append(file_path)
    return img_list


def calculate_fid(src_path, tar_path, pfx, device):
    num_workers = 1
    dims = 2048
    batch_size = 1
    
    src_list = get_img_list(src_path)
    tar_list = get_img_list(tar_path, pfx)
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    m1, s1 = calculate_activation_statistics(src_list, model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(tar_list, model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    src_path = opt.src
    tar_path = opt.tar
    pfx = opt.pfx
    device = opt.device
    

    
    
    
    
    
    

