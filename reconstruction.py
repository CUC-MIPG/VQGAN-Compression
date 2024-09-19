import sys
sys.path.append(".")

# also disable grad to save memory
import torch
import pandas as pd

# Set GPU, run on CPU if not available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import os
import zipfile
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
import torchac
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

# Load config from config_path
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

# Load VQGAN model according to configuration
def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_model(config_path, ckpt_path, is_gumbel=False):
    config = OmegaConf.load(config_path)

    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        sd.pop("loss.discriminator.main.8.weight")
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()
# Normalize the input
def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

# Convert PyTorch tensors to PIL images
def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

# Preprocess PIL images
def preprocess(img):
  # img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
  # img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)
  return img

# Using VQGAN model for image reconstruction
def reconstruct_with_vqgan(x, model):
    with torch.no_grad():
        z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    # using the decoder part of VQModel
    xrec = model.decode(z)
    return xrec,indices,z

#
def reconstruct_decode(z, model):
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  # xrec = transforms.ToPILImage(xrec)
  return xrec

# Stack the images together
def stack_rec(x):
  w, h = x.size[0], x.size[1]
  img = Image.new("RGB", (w, h))
  img.paste(x, (0,0))
  ImageDraw.Draw(img)
  return img

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device='cpu')
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0

def get_uniform_pmf(codebook_size, idx):
    p = 1 / codebook_size[0]
    pmf_L = torch.ones(codebook_size[0]) * p
    pmf_L = pmf_L.repeat(len(idx), 1)

    return pmf_L.to(dtype=torch.float32)


def reconstruct(img, model):
    x_vqgan = preprocess(img)
    x_vqgan = x_vqgan.to(DEVICE)
    print(f"input is of size: {x_vqgan.shape}")
    x0, idx, z = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model)
    img = stack_rec(custom_to_pil(x0[0]))
    return img, idx, z



def compute_bpp_zip(file_path, model, z):
    size = get_file_size_in_bits(file_path)
    print(f'the size of {file_path} is {size} bits')
    # unzip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('unzip/')
        filenames = zip_ref.namelist()
    print("ZIP file decompression successful!")
    print(f"get {filenames} file")

    # load unzip file
    print('unzip/'+filenames[0])
    index = np.load('unzip/'+filenames[0])
    index = torch.tensor(index).to(DEVICE)
    z_ = rearrange(z, 'b c h w -> b h w c').contiguous()
    z_q = model.quantize.embedding(index).view(z_.shape)
    # reshape back to match original input shape
    z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
    x0 = model.decode(z_q)
    rec_img = stack_rec(custom_to_pil(x0[0]))
    rec_img.save('rec/'+filenames[0][:-4]+'_rec.png')
    bpp = size/(x0.shape[2] * x0.shape[3])
    print('bpp:',bpp)
    return bpp


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--logs_path', default='logs/imagenet_f16_16384/', type=str)
    parser.add_argument('--dataset', default='Kodak/', type=str)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    torch.set_grad_enabled(False)

    # Load Model
    config_path = args.logs_path+'configs/model.yaml'
    ckpt_path = args.logs_path+'checkpoints/last.ckpt'
    model = load_model(config_path=config_path, ckpt_path=ckpt_path)

    # set path
    name = args.dataset.replace('/','')
    rec_path = 'rec/' + args.dataset
    if not os.path.exists(rec_path):
        os.makedirs(rec_path)
    index_path = 'index/' + args.dataset
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    tmp_path = 'tmp/' + args.dataset
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # Load Model
    config = load_config(config_path, display=False)
    

    # read image
    img_path = 'data/' + args.dataset
    filenames = os.listdir(img_path)
    file_list = []
    bpp_list = []
    psnr_list = []
    for img in os.listdir(img_path):
        file_list.append(img[:-4])
        image = PIL.Image.open(img_path + img)

        # Caculate the size of image
        num_pixel = image.size[0] * image.size[1]

        # Using VQModel to reconstruct the image
        re_img, index, z = reconstruct(img=image, model=model)

        # save reconstruction image and VQ-Indices
        save_img = rec_path + img
        save_index = index_path + img[:-4]
        re_img.save(save_img)

        # Caculate psnr
        img1 = cv2.imread(img_path + img)
        img2 = cv2.imread(save_img)
        psnr_list.append(psnr(img1, img2))

        # Arithmetic encoding
        idx_cdf_uniform = pmf_to_cdf(get_uniform_pmf(model.quantize.embedding.weight.size(), index))
        byte_stream = torchac.encode_float_cdf(cdf_float=idx_cdf_uniform, sym=index.to(dtype=torch.int16).cpu(),
                                               check_input_bounds=True)

        save_tmp = tmp_path+ img[:-4] +'.b'
        with open(save_tmp, 'wb') as f:
            f.write(byte_stream)

        with open(save_tmp, 'rb') as fin:
            In_byte_stream = fin.read()

        index_out = torchac.decode_float_cdf(idx_cdf_uniform, In_byte_stream)

        assert (index_out.cpu().to(torch.long).equal(index.cpu().to(torch.long))), 'MatchError'

        # index_out = index_out.to(torch.int).to(DEVICE)
        # z_ = rearrange(z, 'b c h w -> b h w c').contiguous()
        # z_q = model.quantize.embedding(index_out).view(z_.shape)
        # # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # x0 = model.decode(z_q)
        # rec_img = stack_rec(custom_to_pil(x0[0]))
        # rec_img.save('rec/' + img[:-4] + '_rec.png')

        num_bits = os.path.getsize(save_tmp) * 8
        bpp = num_bits / num_pixel
        bpp_list.append(bpp)
    average_bpp = sum(bpp_list) / len(bpp_list)
    average_psnr= sum(psnr_list) / len(psnr_list)
    bpp_list.append(average_bpp)
    psnr_list.append(average_psnr)
    file_list.append('Average')
    data = {
        'Image Name': file_list,
        'Bits Per Pixel (BPP)': bpp_list,
        'PSNR Value': psnr_list
    }

    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    output_file = 'bpp/' + name + '.xlsx'
    df.to_excel(output_file, index=False, engine='xlsxwriter')

    print(f'Finish Model:{args.logs_path} test! Avg bpp = {average_bpp} psnr = {average_psnr}')
    print(f'Save bpp.csv to {output_file}')