import glob
import pandas as pd
import torchac
from pytorch_msssim import MS_SSIM
from lpips import LPIPS
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# 放缩到 [-1, 1]
def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def preprocess(img):
    # img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    # img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    with torch.no_grad():
        z, _, [_, _, indices] = model.encode(x)
    # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    # xrec = model.decode(z)
    xrec = x
    return xrec, indices, z


def VQGAN_Encode_forward(x: Image, model):
    input_img = preprocess_vqgan(preprocess(x)).to(DEVICE)
    reconstruction, idx_map, quantize_latent = reconstruct_with_vqgan(input_img, model)

    return reconstruction, idx_map, quantize_latent


def get_uniform_pmf(codebook_size, idx):
    p = 1 / codebook_size[0]
    pmf_L = torch.ones(codebook_size[0]) * p
    pmf_L = pmf_L.repeat(len(idx), 1)

    return pmf_L.to(dtype=torch.float32)


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device='cpu')
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


if __name__ == "__main__":

    # FIXME: 更改数据集与模型参数
    test_dataset_dir_path = 'data/Kodak'
    dataset_name = 'Kodak'

    model_dir = 'logs/kmeans_tune/'
    save_base_dir = 'bpp_uss_torchac'
    bpp_dict = {}

    for model_name in sorted(os.listdir(model_dir)):

        torch.cuda.empty_cache()
        if model_name == '16384_kmeans_4096_epoch' or model_name == '16384_kmeans_8192_epoch':
            continue

        model_base_path = os.path.join(model_dir, model_name)
        ckpt_path = os.path.join(model_base_path, 'epoch1/checkpoints/last.ckpt')
        config_path = os.path.join(model_base_path, 'epoch1/configs/model.yaml')

        model = load_model(config_path=config_path, ckpt_path=ckpt_path)

        # print("orignal:", model.quantize.embedding.weight.size())
        # weight = model.quantize.embedding.weight.cpu() .detach()
        # estimator = KMeans(n_clusters=8)
        # estimator.fit(weight)
        # centroids = estimator.cluster_centers_
        # codebook_2 = nn.Embedding(centroids.shape[0], centroids.shape[1])
        # model.quantize.embedding.weight.data = codebook_2.weight.data.copy_(torch.from_numpy(centroids)).to(DEVICE)

        # FIXME: 数据集的预处理，用Pillow打开即可

        assert os.path.exists(test_dataset_dir_path) and os.path.isdir(test_dataset_dir_path), \
            'Please check your test dataset path!'

        img_paths = sorted(glob.glob(os.path.join(test_dataset_dir_path, '*.*')))

        # FIXME: 调用Encoder，得到index，并做处理，进行熵编码
        # FIXME: torchac将结果编成字节流，官方提供 num_bits = len(字节流) * 8 从数值上来看貌似符合，是否需要写入码流有待商榷

        filename = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]
        bpp_total = []

        for i, img_path in enumerate(img_paths):
            # print(f'Now process: {filename[i]}……')
            img = Image.open(img_path)
            num_pixel = img.size[0] * img.size[1]
            rec, idx, quantize_latent = VQGAN_Encode_forward(img, model=model)
            idx_cdf_uniform = pmf_to_cdf(get_uniform_pmf(model.quantize.embedding.weight.size(), idx))
            byte_stream = torchac.encode_float_cdf(cdf_float=idx_cdf_uniform, sym=idx.to(dtype=torch.int16).cpu(),
                                                   check_input_bounds=True)

            with open(f'tmp/{filename[i]}.b', 'wb') as f:
                f.write(byte_stream)

            with open(f'tmp/{filename[i]}.b', 'rb') as fin:
                In_byte_stream = fin.read()

            idx_out = torchac.decode_float_cdf(idx_cdf_uniform, In_byte_stream)
            assert (idx_out.cpu().to(torch.long).equal(idx.cpu().to(torch.long))), 'MatchError'

            num_bits = os.path.getsize(f'tmp/{filename[i]}.b') * 8
            bpp = num_bits / num_pixel
            bpp_total.append(bpp)
            # rec = custom_to_pil(rec[0])
            # rec.save(os.path.join(os.path.join(save_base_dir, model_name, dataset_name), f'{filename[i]}.png'))

        # FIXME: 数据处理

        df = pd.DataFrame([filename, bpp_total]).T
        df.columns = ['input image', 'bpp']
        save_dir = os.path.join(save_base_dir, model_name, dataset_name)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        df_path = os.path.join(save_dir, 'bpp.xlsx')
        df.to_excel(df_path)

        bpp_dict[model_name] = sum(bpp_total) / len(filename)

        print(f'Finish Model:{model_name} test! Avg bpp = {sum(bpp_total) / len(filename)}')
        print(f'Save bpp.xlsx to {df_path}')

    bpp_dict = sorted(bpp_dict.items(), key=lambda x: x[1])
    print(bpp_dict)
