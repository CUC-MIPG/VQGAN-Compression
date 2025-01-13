import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid

from taming.util import instantiate_from_config
from taming.modules.util import SOSProvider
from taming.modules.transformer.mingpt import GPT

from taming.modules.entropy_coding import (
    encoding_decoding_torchac,
    encoding_masked_torchac
)
from taming.util import compute_padding


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class GPT_addtoken(GPT):
    def __init__(self, 
                 vocab_size, block_size, n_layer=12, n_head=8, n_embd=256, 
                 embd_pdrop=0, resid_pdrop=0, attn_pdrop=0, n_unmasked=0,
                 extra_token=0
                 ):
        super().__init__(vocab_size, block_size, n_layer, n_head, n_embd, 
                         embd_pdrop, resid_pdrop, attn_pdrop, n_unmasked)
        self.tok_emb = nn.Embedding(vocab_size + extra_token, n_embd)
        

class Transformer_stage0(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 trainging_config,
                 test_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 ):
        super().__init__()
        # load transformer and codec model
        # image
        self.first_stage_key = first_stage_key
        # transformer_config:
        #       target: taming.models.transformer_4step.GPT_addtoken
        #       params:
        #         vocab_size: 16
        #         block_size: 256
        #         n_layer: 32
        #         n_head: 16
        #         n_embd: 1152
        #         extra_token: 1
        # self.transformer = transformer_4step.GPT_addtoken()
        self.transformer = instantiate_from_config(config=transformer_config)

        # first_stage_config:
        #     target: taming.models.vqgan.VQModel
        #     params:
        #     ckpt_path:./ pretrain / kmeans_tune / 16384
        #     _kmeans_32_epoch / epoch1 / checkpoints / last.ckpt
        #     embed_dim: 256
        #     n_embed: 32
        #     ddconfig:
        #     double_z: false
        #     z_channels: 256
        #     resolution: 256
        #     in_channels: 3
        #     out_ch: 3
        #     ch: 128
        #     ch_mult:
        #     - 1
        #     - 1
        #     - 2
        #     - 2
        #     - 4
        #     num_res_blocks: 2
        #     attn_resolutions:
        #     - 16
        #     dropout: 0.0
        #
        #     lossconfig:
        #         target: taming.modules.losses.DummyLoss
        # def init_first_stage_from_ckpt(self, config):
        #     model = instantiate_from_config(config)
        #     model = model.eval()
        #     model.train = disabled_train
        #     self.first_stage_model = model
        self.init_first_stage_from_ckpt(first_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        
        # output sane index shape
        self.first_stage_model.quantize.sane_index_shape = True
        
        # load ckpt
        # def init_from_ckpt(self, path, ignore_keys=list()):
        #     sd = torch.load(path, map_location="cpu")["state_dict"]
        #     for k in sd.keys():
        #         for ik in ignore_keys:
        #             if k.startswith(ik):
        #                 self.print("Deleting key {} from state_dict.".format(k))
        #                 del sd[k]
        #     self.load_state_dict(sd, strict=False)
        #     print(f"Restored from {path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # extra start token
        # vocab_size: 32
        self.start_token_idx = transformer_config.params.vocab_size

        # init training and test param
        # trainging_config:
        #     use_simple_optim: true
        #     init_learning_rate: 1.0e-4
        #     lr_milestone:
        #     - 120000
        #     - 180000
        #     - 240000
        #     lr_milestones_gamma: 0.5
        #     scheduler_interval: step
        self.init_training_param(trainging_config)
        #
        self.test_config = test_config
        if monitor is not None:
            self.monitor = monitor


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

            
    def init_training_param(self, config):
        if config:
            # optim
            self.init_learning_rate = config.init_learning_rate
            self.lr_milestone = config.lr_milestone
            self.lr_milestones_gamma = config.lr_milestones_gamma
            self.scheduler_interval = config.scheduler_interval
            assert self.scheduler_interval in ['step', 'epoch']
        else:
            # optim
            self.init_learning_rate = 1e-4
            self.lr_milestone = [1,2]
            self.lr_milestones_gamma = 0.5
            self.scheduler_interval = 'epoch'


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.transformer.parameters(), 
                lr=self.init_learning_rate
            )
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=self.lr_milestone, 
                    gamma=self.lr_milestones_gamma
                ),
                'interval': self.scheduler_interval,
                'frequency': 1,
            }
        }
        return optim_dict
    
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 1)
        
    def forward(self, x):
        # one step to produce the logits
        B, C, H, W = x.shape
        _, z_indices = self.encode_to_z(x)
        z_indices = self.extract_position0(z_indices)
        # z_indices: torch.Size([1, 256])
        z_indices = z_indices.reshape(B, -1)

        # concat masked indicies with dummy condition index
        # tensor([[1]], device='cuda:1')
        start_token = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        # self.start_token_idx: 32
        # tensor([[32]], device='cuda:1')
        start_token = start_token * self.start_token_idx
        print(start_token.shape)
        print('z_indices[:, :-1]:',z_indices[:, :-1].shape)
        input_indices = torch.cat([start_token, z_indices[:, :-1]], dim=1)
        print('input_indices:',input_indices.shape)
        print(z_indices,input_indices)
        exit()
        # make the prediction
        logits, _ = self.transformer(input_indices)
        target = z_indices
        loss =  F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        return logits, target, loss


    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2]
        indices = self.permuter(indices)
        return quant_z, indices
    
    
    def extract_position0(self, indices):
        return indices[:, ::2, ::2]

    
    def get_img(self, batch, N=None):
        x = batch[self.first_stage_key]
        if N is not None:
            x = x[:N]
        return x
    
    
    def common_step(self, batch, batch_idx):
        img = self.get_img(batch)
        _, _, loss = self.forward(img)
        return loss
    
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        test_config = self.test_config
        self.eval()
        
        img = self.get_img(batch)
        B, _, H, W = img.shape
        assert B == 1           # only process batch == 1!
        pix_num = B * H * W     # pix number of a image
        
        # get window length
        block_size = self.transformer.config.block_size
        window_size = int(math.sqrt(block_size))
        assert math.sqrt(block_size) % 1.0 == 0.0, "block size must be int(x) ** 2!"
        
        # encode image
        _, z_indices = self.encode_to_z(img)
        z_indices = self.extract_position0(z_indices)
        B_z, H_indices, W_indices = z_indices.shape
        
        # probs container
        information = torch.zeros([B, H_indices, W_indices], device=device, dtype=torch.float32)
        
        # concat masked indicies with dummy condition index
        start_token = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        start_token = start_token * self.start_token_idx
        
        # loop over z_masked
        for h in range(H_indices):
            for w in range(W_indices):
                # extract window, 16x16, as paper does
                if h == 0:
                    window_left = max(0, w - (window_size - 1))
                    window_right = window_left + (window_size - 1)
                else:
                    window_left = min(max(0, w - (window_size // 2)), W_indices - window_size)
                    window_right = window_left + window_size
                w_top = min(max(0, h - (window_size // 2)), H_indices - window_size)
                w_bottom = w_top + window_size

                # ---------- extract pixels like gated pixel CNN ----------
                # support pixels above current pixel
                masked_support_indices_vec = z_indices[:, w_top:h, window_left:window_right].reshape(B, -1)
                # support pixels in the left of current pixel
                masked_support_indices_hor = z_indices[:, h, window_left:w].reshape(B, -1)
                # flatten pixels
                masked_support_indices = torch.cat(
                    [start_token, masked_support_indices_vec, masked_support_indices_hor], 
                    dim=1
                ).view(B, -1)
                
                # calculate probability
                logits, _ = self.transformer(masked_support_indices)
                logits = logits[:, -1, :]                   # extract logits for current location
                probs = F.softmax(logits, dim=-1).view(B, -1)
                information[:, h, w] = -1. * torch.log2(probs[:, z_indices[:, h, w].item()])
        
        # calculate entropy in position 0
        entropy_bpp_0 = information.sum().item() / pix_num
        
        # result dict overall
        result_dict = {
            # log encoding results
            "position_0_bpp": entropy_bpp_0,
        }
        
        # image dict
        img_dict = { }
        
        return {
            'loss': 0.0,
            'result': result_dict,
            'imgs': img_dict
        }
    


class Transformer_stage1_parallel(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 trainging_config,
                 test_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 ):
        super().__init__()
        
        # extra start token
        extra_token_idx = [
            transformer_config.params.vocab_size + i
            for i in range(4)
        ]
        self.mask_token_idx = extra_token_idx[0]
        self.split_token_idx = extra_token_idx[1:]
        if transformer_config.params.extra_token < 4:
            transformer_config.params.extra_token = 4
            print("\n[WARNING!!!] invalid extra_token num!\n")
        
        # load transformer and codec model
        self.first_stage_key = first_stage_key
        
        self.transformer = instantiate_from_config(config=transformer_config)
        self.init_first_stage_from_ckpt(first_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.codebook_size = self.first_stage_model.quantize.n_e
        
        # output sane index shape
        self.first_stage_model.quantize.sane_index_shape = True
        
        # load ckpt
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # init training and test param
        self.init_training_param(trainging_config)
        if monitor is not None:
            self.monitor = monitor
            
        # init test config
        self.test_config = test_config
        if test_config:
            self.transformer_postion0 = instantiate_from_config(
            config=test_config.transformer_position0_config)
            sd_path = test_config.transformer_position0_config["ckpt_path"]
            sd = torch.load(sd_path, map_location="cpu")["state_dict"]
            sd_new = {}
            for k, v in sd.items():
                if k.startswith("transformer"):
                    k = k.replace("transformer.", "")
                    sd_new[k] = v
            self.transformer_postion0.load_state_dict(sd_new, strict=True)
            
            self.mask_model = instantiate_from_config(test_config.mask_model).eval()
            self.quality_model = instantiate_from_config(test_config.quality_model).eval()


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

            
    def init_training_param(self, config):
        if config:
            # optim
            self.init_learning_rate = config.init_learning_rate
            self.lr_milestone = config.lr_milestone
            self.lr_milestones_gamma = config.lr_milestones_gamma
            self.scheduler_interval = config.scheduler_interval
            assert self.scheduler_interval in ['step', 'epoch']
            self.mask_ratio = config.mask_ratio
            self.p_mask = config.p_mask
            self.val_avg_times = config.val_avg_times
        else:
            # optim
            self.init_learning_rate = 1e-4
            self.lr_milestone = [1,2]
            self.lr_milestones_gamma = 0.5
            self.scheduler_interval = 'epoch'
            self.mask_ratio = [0.0, 0.75]
            self.p_mask = 0.9
            self.val_avg_times = 5
        self.mask_ratio_range = self.mask_ratio[1] - self.mask_ratio[0]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.transformer.parameters(), 
                lr=self.init_learning_rate
            )
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=self.lr_milestone, 
                    gamma=self.lr_milestones_gamma
                ),
                'interval': self.scheduler_interval,
                'frequency': 1,
            }
        }
        return optim_dict
    
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 1)
    
    @torch.no_grad()
    def generate_mask(self, x, mask_ratio):
        ''' change by xnf'''
        if len(x.shape) == 2:   # when input is [B, L]
            B, L = x.shape      # batch, length
            reshape = False
        elif len(x.shape) == 3: # when input is [B, H, W]
            B, H, W = x.shape
            L = H * W
            reshape = True
        else:
            raise NotImplementedError("Invalid input shape!")
            
        len_masked = int(L * mask_ratio)
        
        # random noise
        noise = torch.rand(B, L, device=x.device)
        
        # permute noise tensor, small to large
        ids_permute = torch.argsort(noise, dim=1)       # small to large
        ids_th = ids_permute[:, len_masked].unsqueeze(-1) # find the threshold
        th = torch.gather(noise, dim=1, index=ids_th)   # threshold value
        
        if reshape:
            noise = noise.reshape([B, H, W])
        
        return noise.ge(th)     # generate mask, noise values < threshold will be masked
    
    
    @torch.no_grad()
    def apply_mask(self, x, mask_ratio, mask_token=None):
        # generate mask
        mask = self.generate_mask(x, mask_ratio)
        inv_mask = mask == False
        
        if not mask_token:
            mask_token = self.mask_token_idx
        mask_map = torch.ones_like(x) * mask_token

        x_masked = x * mask.int() + mask_map * inv_mask.int()
        return x_masked, mask, inv_mask

        
    def forward(self, x, mask_ratio=0.0):
        # one step to produce the logits
        B, C, H, W = x.shape
        _, z_indices, dist = self.encode_to_z(x)
        z_groups = self.extract_groups(z_indices, B)
        
        # add mask to position 1 and 2
        ones = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        z_masked_input = torch.cat([
            ones * self.split_token_idx[0],
            z_groups[0],
            ones * self.split_token_idx[1],
            self.apply_mask(z_groups[1], mask_ratio)[0],
            ones * self.split_token_idx[2],
            self.apply_mask(z_groups[2], mask_ratio)[0],
        ], dim=1)
        
        # target, is position 1, 2 and 3
        z_target = torch.cat([z_groups[i] for i in range(1,4)], dim=1)
        
        # make the prediction
        logits, _ = self.transformer(z_masked_input)
        
        # split prediction and remove unused token
        logits_split = torch.chunk(logits, chunks=3, dim=1)
        logits_remove_extra = torch.cat([logits_split[i][:, :-1] for i in range(3)], dim=1)
        # probs_remove_unused = F.softmax(logits_remove_unused, dim=2)
        
        loss = F.cross_entropy(logits_remove_extra.reshape(-1, logits_remove_extra.size(-1)), z_target.reshape(-1))
        
        # # soft labeling
        # dist_groups = self.extract_groups(dist, shape=(B, -1, dist.shape[-1]))
        # dist_target = torch.cat([dist_groups[i] for i in range(1,4)], dim=1)
        # soft_label_logits = torch.exp(-1.0 * dist_target / self.tau)
        # soft_label_probs = F.softmax(soft_label_logits, dim=2)
        
        # calculate loss
        # loss_cross_entropy = F.cross_entropy(logits_remove_unused.reshape(-1, logits_remove_unused.size(-1)), z_target.reshape(-1))
        # loss_soft_label = F.l1_loss(probs_remove_unused, soft_label_probs)
        # loss = self.alpha * loss_cross_entropy + self.beta * loss_soft_label
        # loss_info = {
        #     "loss_cross_entropy": loss_cross_entropy,
        #     "loss_soft_label": loss_soft_label
        # }
        
        return logits_split, loss


    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2]
        indices = self.permuter(indices)
        dist = info[3]
        return quant_z, indices, dist
    
    
    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x
    
    
    def extract_groups(self, x, B=None, shape=None):
        assert shape or B, "Must provide batch dim or shape!"
        if not shape:
            shape = (B, -1)
        return (
            x[:, 0::2, 0::2].reshape(shape), 
            x[:, 1::2, 1::2].reshape(shape), 
            x[:, 1::2, 0::2].reshape(shape),
            x[:, 0::2, 1::2].reshape(shape),
        )

    
    def get_img(self, batch, N=None):
        x = batch[self.first_stage_key]
        if N is not None:
            x = x[:N]
        return x
    
    
    def common_step(self, batch, batch_idx, mask_ratio=0.0):
        img = self.get_img(batch)
        _, loss = self.forward(img, mask_ratio)
        return loss
    
    
    def training_step(self, batch, batch_idx):
        # generate random mask ratio
        if torch.rand(1) < self.p_mask:
            mask_ratio = torch.rand(1) * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
            mask_ratio = mask_ratio.item()
        else:
            mask_ratio = 0.0
        
        loss = self.common_step(batch, batch_idx, mask_ratio)
        self.log("train/loss", loss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("train/cross_entropy_loss", loss_info["loss_cross_entropy"], 
        #          prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # self.log("train/soft_label_loss", loss_info["loss_soft_label"], 
        #          prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        mask_ratio = 0.0
        loss_no_mask = self.common_step(batch, batch_idx, mask_ratio)
        self.log("val/loss_no_mask", loss_no_mask, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        avg_times = 10
        mask_ratio = (self.mask_ratio[1] - self.mask_ratio[0]) / 2.0
        loss_mask = 0.0
        for i in range(avg_times):
            loss_mask += self.common_step(batch, batch_idx, mask_ratio)
        loss_mask /= avg_times
        self.log("val/loss_with_mask", loss_mask, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        loss_avg = (loss_no_mask + loss_mask) / 2
        self.log("val/loss", loss_avg, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/cross_entropy_loss", loss_info["loss_cross_entropy"], 
        #          prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/soft_label_loss", loss_info["loss_soft_label"], 
        #          prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss_avg
            

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        test_config = self.test_config
        self.eval()
        
        # def functions
        def top_k_logits(logits, k):
            assert len(logits.shape) == 3, "input logits must be [B, L, C]!"
            v, ix = torch.topk(logits, k, dim=2)
            out = logits.clone()
            out[out < v[..., [-1]]] = -float('Inf')
            return out
        
        def get_probs_prediction(z_input, chunks, window_size):
            logits, _ = self.transformer(z_input)
            logits = torch.chunk(logits, chunks=chunks, dim=1)[-1][:, :-1]
            probs = F.softmax(logits, dim=2)
            
            B, L, C = probs.shape
            logits_topk = top_k_logits(logits, test_config.top_k)
            probs_topk = F.softmax(logits_topk, dim=2).reshape(B * L, C)
            ix = torch.multinomial(probs_topk, num_samples=1).reshape(B, L)
            
            # reshape output
            probs = probs.reshape(B, window_size // 2, window_size // 2, C)
            ix = ix.reshape(B, window_size // 2, window_size // 2)
            return probs, ix
        
        # load image
        img = self.get_img(batch)
        B, _, H, W = img.shape
        assert B == 1           # only process batch == 1!
        pix_num = B * H * W     # pix number of a image
        
        # pad img
        pad, unpad = compute_padding(H, W, min_div=self.downsampling_factor)
        img = F.pad(img, pad, mode='replicate')
        
        # get window length
        block_size = self.transformer.config.block_size
        window_size = int(math.sqrt((block_size - 3) // 3)) * 2
        assert window_size % 1.0 == 0.0, "window size must be int(x) ** 2!"
        
        # encode image
        quant_z, z_indices, _ = self.encode_to_z(img)
        B_z, H_size, W_size = z_indices.shape
        
        # generate and apply mask
        mask, mask_info = self.mask_model(img, z_indices)
        inv_mask = (mask == 0).int()
        z_masked = z_indices * mask + self.mask_token_idx * inv_mask
        
        # mask bpp
        mask_bpp = mask_info["byte_mask"] * 8 / pix_num
        mask_compressed_bpp = mask_info["byte_compressed_mask"] * 8 / pix_num
        
        # containers
        ones = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        pmf_container = torch.zeros([B, H_size, W_size, self.codebook_size], device=device, dtype=torch.float32)
        prediction_container = torch.zeros([B, H_size, W_size], device=device, dtype=torch.int32)
        used_container = torch.zeros([B, H_size, W_size], device=device, dtype=torch.bool)
        
        # define used_area_size
        used_size = self.test_config.used_size
        assert used_size % 2 == 0
        border_size = (window_size - used_size) // 2
        
        # calculate steps
        h_step_num = math.ceil((H_size - border_size * 2) / used_size)
        w_step_num = math.ceil((W_size - border_size * 2) / used_size)
        
        # loop over z_indices, for step 1,2,3
        for h_step in range(h_step_num):
            for w_step in range(w_step_num):
                
                # calculate window position
                window_bottom = min(window_size + used_size * h_step, H_size)
                window_top = window_bottom - window_size
                assert window_top >= 0
                window_right = min(window_size + used_size * w_step, W_size)
                window_left = window_right - window_size
                assert window_left >= 0
                
                # create window container
                pmf_window = torch.zeros([B, window_size, window_size, self.codebook_size], device=device, dtype=torch.float32)
                prediction_window = torch.zeros([B, window_size, window_size], device=device, dtype=torch.int32)
                
                # extract window and split into groups
                z_window = z_masked[:, window_top:window_bottom, window_left:window_right]
                z_window_grouped = self.extract_groups(z_window, B)
                
                # step 1
                z_input = torch.cat([
                    ones * self.split_token_idx[0],
                    z_window_grouped[0],
                ], dim=1)
                pmf_s1, ix_s1 = get_probs_prediction(z_input, chunks=1, window_size=window_size)
                pmf_window[:, 1::2, 1::2] = pmf_s1
                prediction_window[:, 1::2, 1::2] = ix_s1
                
                # step 2
                z_input = torch.cat([
                    ones * self.split_token_idx[0],
                    z_window_grouped[0],
                    ones * self.split_token_idx[1],
                    z_window_grouped[1],
                ], dim=1)
                pmf_s2, ix_s2 = get_probs_prediction(z_input, chunks=2, window_size=window_size)
                pmf_window[:, 1::2, 0::2] = pmf_s2
                prediction_window[:, 1::2, 0::2] = ix_s2
                
                # step 3
                z_input = torch.cat([
                    ones * self.split_token_idx[0],
                    z_window_grouped[0],
                    ones * self.split_token_idx[1],
                    z_window_grouped[1],
                    ones * self.split_token_idx[2],
                    z_window_grouped[2],
                ], dim=1)
                pmf_s3, ix_s3 = get_probs_prediction(z_input, chunks=3, window_size=window_size)
                pmf_window[:, 0::2, 1::2] = pmf_s3
                prediction_window[:, 0::2, 1::2] = ix_s3
                
                # write pmf and prediction
                is_used = used_container[:, window_top:window_bottom, window_left:window_right]
                input_mask = ~ is_used        # if is_used == 0, this location is not used, we write value into there.
                # for center area, we force to write value into it.
                input_mask[:, border_size:window_size-border_size, border_size:window_size-border_size] = 1
                pmf_container[:, window_top:window_bottom, window_left:window_right] = \
                    pmf_window * input_mask.unsqueeze(-1).float() + \
                    pmf_container[:, window_top:window_bottom, window_left:window_right] * (~input_mask).unsqueeze(-1).float()
                prediction_container[:, window_top:window_bottom, window_left:window_right] = \
                    prediction_window * input_mask.int() +\
                    prediction_container[:, window_top:window_bottom, window_left:window_right] * (~input_mask).int()
                
                # after write, we mark this part to be used
                used_container[:, window_top:window_bottom, window_left:window_right] = 1
                
        # prediction step 0
        z_step0 = z_indices[:, 0::2, 0::2]
        pmf_step0 = torch.zeros([B, H_size // 2, W_size // 2, self.codebook_size], device=device, dtype=torch.float32)
        block_size_s0 = self.transformer_postion0.config.block_size
        window_size_s0 = int(math.sqrt(block_size_s0))
        assert window_size_s0 % 1.0 == 0.0
        start_token = ones * self.mask_token_idx  # we use the mask token as the start token in step 0 transformer.
        
        # loop over z_indices, for step 0
        for h_step in range(H_size // 2):
            for w_step in range(W_size // 2):
                # extract window, 16x16, as paper does
                if h_step == 0:
                    window_left = max(0, w_step - (window_size_s0 - 1))
                    window_right = window_left + (window_size_s0 - 1)
                else:
                    window_left = min(max(0, w_step - (window_size_s0 // 2)), W_size - window_size)
                    window_right = window_left + window_size_s0
                window_top = min(max(0, h_step - (window_size_s0 // 2)), H_size - window_size_s0)
                window_bottom = window_top + window_size_s0

                # ---------- extract pixels like gated pixel CNN ----------
                # support pixels above current pixel
                support_indices_vec = z_step0[:, window_top:h_step, window_left:window_right].reshape(B, -1)
                # support pixels in the left of current pixel
                support_indices_hor = z_step0[:, h_step, window_left:w_step].reshape(B, -1)
                # flatten pixels
                masked_support_indices = torch.cat(
                    [start_token, support_indices_vec, support_indices_hor], 
                    dim=1
                ).view(B, -1)
                
                # calculate probability
                logits, _ = self.transformer_postion0(masked_support_indices)
                logits = logits[:, -1, :]                   # extract logits for current location
                probs = F.softmax(logits, dim=-1).view(B, -1)
                pmf_step0[:, h_step, w_step] = probs
                
        # write step 0 to container
        pmf_container[:, 0::2, 0::2] = pmf_step0
        prediction_container[:, 0::2, 0::2] = z_indices[:, 0::2, 0::2]
        
        # fill prediction into indices
        z_predicted = z_indices * mask.int() + prediction_container * inv_mask.int()
        
        # decode image
        img_rec = self.decode_to_img(z_indices, quant_z.shape).clamp(-1.0, 1.0)
        img_pred_rec = self.decode_to_img(z_predicted, quant_z.shape).clamp(-1.0, 1.0)
        
        # unpad x
        img = F.pad(img, unpad)
        img_rec = F.pad(img_rec, unpad)
        img_pred_rec = F.pad(img_pred_rec, unpad)
        
        # rescale to [0,1]
        img_ori = img / 2.0 + 0.5           
        img_rec = img_rec / 2.0 + 0.5
        img_pred_rec = img_pred_rec / 2.0 + 0.5
        
        # calculate quality
        quality_dict_rec = self.quality_model(img_ori, img_rec)
        quality_dict_pred = self.quality_model(img_ori, img_pred_rec)
        
        # perform entropy coding
        byte_stream, byte_num, information = encoding_masked_torchac(
            z_indices.to(dtype=torch.int16),
            pmf_container.to(dtype=torch.float32),
            mask
        )
        enc_bpp = byte_num * 8 / pix_num     # calculate bpp after entropy coding
        entropy_bpp = information / pix_num
        total_compressed_bpp = enc_bpp + mask_compressed_bpp

        # log encoding results
        self.log("test/enc_bpp", enc_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/entropy_bpp", entropy_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/mask_bpp", mask_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/mask_compressed_bpp", mask_compressed_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/total_compressed_bpp", total_compressed_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log predicted quality results
        self.log("test/pred_psnr", quality_dict_pred['psnr'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_msssim", quality_dict_pred['msssim'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_lpips", quality_dict_pred['lpips'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_dists", quality_dict_pred['dists'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log recon quality results
        self.log("test/rec_psnr", quality_dict_rec['psnr'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_msssim", quality_dict_rec['msssim'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_lpips", quality_dict_rec['lpips'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_dists", quality_dict_rec['dists'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log images
        mask_large = mask.repeat(3, 1, 1).view(B, 3, H_size, W_size)
        mask_large = F.interpolate(mask_large.float(), size=[H, W], mode='nearest')
        log_imgs = torch.cat([img_ori, img_rec, mask_large, img_pred_rec], dim=0)
        log_imgs = make_grid(log_imgs, padding=10)
        self.logger.experiment.add_image("test/imgs", log_imgs, batch_idx)
        
        # result dict overall
        result_dict = {
            # log encoding results
            "enc_bpp": enc_bpp,
            "entropy_bpp": entropy_bpp,
            "mask_bpp": mask_bpp,
            "mask_compressed_bpp": mask_compressed_bpp,
            "total_compressed_bpp": total_compressed_bpp,
            
            # log predicted quality results
            "PSNR_pred": quality_dict_pred['psnr'],
            "MS-SSIM_pred": quality_dict_pred['msssim'],
            "LPIPS_pred": quality_dict_pred['lpips'],
            "DISTS_pred": quality_dict_pred['dists'],
            
            # log predicted quality results
            "PSNR_rec": quality_dict_rec['psnr'],
            "MS-SSIM_rec": quality_dict_rec['msssim'],
            "LPIPS_rec": quality_dict_rec['lpips'],
            "DISTS_rec": quality_dict_rec['dists'],
        }
        
        # image dict
        img_dict = {
            "original": img_ori,
            "reconstruction": img_rec,
            "predicted": img_pred_rec,
            "mask": mask_large,
            "concated": log_imgs
        }
        
        return {
            'loss': 0.0,
            'result': result_dict,
            'imgs': img_dict
        }
        
        

class Transformer_stage1_serial(Transformer_stage1_parallel):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 trainging_config,
                 test_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 ):
        super().__init__(
                 transformer_config,
                 first_stage_config,
                 trainging_config=None,
                 test_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
        )
        
        # extra start token
        extra_token_idx = [
            transformer_config.params.vocab_size + i
            for i in range(5)
        ]
        self.mask_token_idx = extra_token_idx[0]
        self.split_token_idx = extra_token_idx[1:]
        if transformer_config.params.extra_token < 5:
            transformer_config.params.extra_token = 5
            print("\n[WARNING!!!] invalid extra_token num!\n")
        
        # load transformer and codec model
        self.first_stage_key = first_stage_key
        
        self.transformer = instantiate_from_config(config=transformer_config)
        self.init_first_stage_from_ckpt(first_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.codebook_size = self.first_stage_model.quantize.n_e
        
        # output sane index shape
        self.first_stage_model.quantize.sane_index_shape = True
        
        # load ckpt
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # init training and test param
        self.init_training_param(trainging_config)
        self.test_config = test_config
        if monitor is not None:
            self.monitor = monitor
            
        # init test config
        self.test_config = test_config
        if test_config:
            self.mask_model = instantiate_from_config(test_config.mask_model).eval()
            self.quality_model = instantiate_from_config(test_config.quality_model).eval()

        
    def forward(self, x, mask_ratio=0.0):
        # one step to produce the logits
        B, C, H, W = x.shape
        _, z_indices, dist = self.encode_to_z(x)
        z_groups = self.extract_groups(z_indices, B)
        
        # add mask to position 1 and 2
        ones = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        z_masked_input = torch.cat([
            ones * self.split_token_idx[0],
            z_groups[0],
            ones * self.split_token_idx[1],
            self.apply_mask(z_groups[1], mask_ratio)[0],
            ones * self.split_token_idx[2],
            self.apply_mask(z_groups[2], mask_ratio)[0],
            ones * self.split_token_idx[3],
            self.apply_mask(z_groups[3], mask_ratio)[0],
        ], dim=1)
        
        # target, is position 1, 2 and 3
        z_target = torch.cat([z_groups[i] for i in range(0,4)], dim=1)
        
        # make the prediction
        logits, _ = self.transformer(z_masked_input)
        
        # split prediction and remove unused token
        logits_split = torch.chunk(logits, chunks=4, dim=1)
        logits_remove_extra = torch.cat([logits_split[i][:, :-1] for i in range(0, 4)], dim=1)
        
        loss = F.cross_entropy(logits_remove_extra.reshape(-1, logits_remove_extra.size(-1)), z_target.reshape(-1))
        
        return logits_split, loss


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        test_config = self.test_config
        self.eval()
        torch.cuda.empty_cache()
        
        # def functions
        def top_k_logits(logits, k):
            v, ix = torch.topk(logits, k, dim=-1)
            out = logits.clone()
            out[out < v[..., [-1]]] = -float('Inf')
            return out
        
        def loop_in_window(step):
            pmf_container = torch.zeros([B, H_size_half, W_size_half, self.codebook_size], device=device, dtype=torch.float32)
            prediction_container = torch.zeros([B, H_size_half, W_size_half], device=device, dtype=torch.int32)
            
            for h_step in range(h_step_num):
                for w_step in range(w_step_num):
                    # extract window, as paper does
                    if h_step == 0:
                        window_left = max(0, w_step - (window_size - 1))
                        window_right = window_left + (window_size - 1)
                    else:
                        window_left = min(max(0, w_step - (window_size // 2)), W_size_half - window_size)
                        window_right = window_left + window_size
                    window_top = min(max(0, h_step - (window_size // 2)), H_size_half - window_size)
                    window_bottom = window_top + window_size
                    
                    window_left_1 = min(max(0, w_step - (window_size // 2)), W_size_half - window_size)
                    window_right_1 = window_left_1 + window_size
                    
                    if step == 0:
                        support_indices_last = torch.cat([
                            ones * self.split_token_idx[0],
                        ], dim=1)
                    elif step == 1:
                        support_indices_last = torch.cat([
                            ones * self.split_token_idx[0],
                            z_masked_group[0][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[1],
                        ], dim=1)
                    elif step == 2:
                        support_indices_last = torch.cat([
                            ones * self.split_token_idx[0],
                            z_masked_group[0][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[1],
                            z_masked_group[1][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[2],
                        ], dim=1)
                    elif step == 3:
                        support_indices_last = torch.cat([
                            ones * self.split_token_idx[0],
                            z_masked_group[0][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[1],
                            z_masked_group[1][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[2],
                            z_masked_group[2][:, window_top:window_bottom, window_left_1:window_right_1].reshape(B, -1),
                            ones * self.split_token_idx[3],
                        ], dim=1)
                    else:
                        raise NotImplementedError()
                    
                    support_indices_vec = z_masked_group[step][:, window_top:h_step, window_left:window_right].reshape(B, -1)
                    support_indices_hor = z_masked_group[step][:, h_step, window_left:w_step].reshape(B, -1)
                    masked_support_indices = torch.cat(
                        [support_indices_last, support_indices_vec, support_indices_hor], 
                        dim=1
                    ).view(B, -1)
                    
                    # calculate probability
                    logits, _ = self.transformer(masked_support_indices)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=1).reshape(B, -1)
                    pmf_container[:, h_step, w_step] = probs
                    
                    # make prediction
                    logits_topk = top_k_logits(logits, test_config.top_k)
                    probs_topk = F.softmax(logits_topk, dim=1).reshape(B, -1)
                    ix = torch.multinomial(probs_topk, num_samples=1).reshape(B)
                    prediction_container[:, h_step, w_step] = ix
            
            return pmf_container, prediction_container
                    
        # load image
        img = self.get_img(batch)
        B, _, H, W = img.shape
        assert B == 1           # only process batch == 1!
        pix_num = B * H * W     # pix number of a image
        
        # pad img
        pad, unpad = compute_padding(H, W, min_div=self.downsampling_factor)
        img = F.pad(img, pad, mode='replicate')
        
        # get window length
        block_size = self.transformer.config.block_size
        window_size = int(math.sqrt(block_size - 4)) // 2
        assert window_size % 1.0 == 0.0, "window size must be int(x) ** 2!"
        
        # encode image
        quant_z, z_indices, _ = self.encode_to_z(img)
        B_z, H_size, W_size = z_indices.shape
        H_size_half, W_size_half = H_size // 2, W_size // 2
        torch.cuda.empty_cache()
        
        # generate and apply mask
        mask, mask_info = self.mask_model(img, z_indices)
        inv_mask = (mask == 0).int()
        z_masked = z_indices * mask + self.mask_token_idx * inv_mask
        
        # split group
        z_masked_group = self.extract_groups(z_masked, shape=[B, H_size // 2, W_size // 2])
        
        # mask bpp
        mask_bpp = mask_info["byte_mask"] * 8 / pix_num
        mask_compressed_bpp = mask_info["byte_compressed_mask"] * 8 / pix_num
        
        # containers
        ones = torch.ones([B, 1], device=z_indices.device, dtype=z_indices.dtype)
        pmf_container = torch.zeros([B, H_size, W_size, self.codebook_size], device=device, dtype=torch.float32)
        prediction_container = torch.zeros([B, H_size, W_size], device=device, dtype=torch.int32)
        
        # calculate steps
        h_step_num = H_size // 2
        w_step_num = W_size // 2
        
        # stage 0
        _pmf, _prediction = loop_in_window(step=0)
        pmf_container[:, ::2, ::2] = _pmf
        prediction_container[:, ::2, ::2] = z_indices[:, ::2, ::2]
        
        # stage 1
        _pmf, _prediction = loop_in_window(step=1)
        pmf_container[:, 1::2, 1::2] = _pmf
        prediction_container[:, 1::2, 1::2] = _prediction
        
        # stage 2
        _pmf, _prediction = loop_in_window(step=2)
        pmf_container[:, 1::2, ::2] = _pmf
        prediction_container[:, 1::2, ::2] = _prediction
        
        # stage 3
        _pmf, _prediction = loop_in_window(step=3)
        pmf_container[:, ::2, 1::2] = _pmf
        prediction_container[:, ::2, 1::2] = _prediction
        
        # fill prediction into indices
        z_predicted = z_indices * mask.int() + prediction_container * inv_mask.int()
        
        # decode image
        torch.cuda.empty_cache()
        img_rec = self.decode_to_img(z_indices, quant_z.shape).clamp(-1.0, 1.0)
        img_pred_rec = self.decode_to_img(z_predicted, quant_z.shape).clamp(-1.0, 1.0)
        
        # unpad x
        img = F.pad(img, unpad)
        img_rec = F.pad(img_rec, unpad)
        img_pred_rec = F.pad(img_pred_rec, unpad)
        
        # rescale to [0,1]
        img_ori = img / 2.0 + 0.5           
        img_rec = img_rec / 2.0 + 0.5
        img_pred_rec = img_pred_rec / 2.0 + 0.5
        
        # calculate quality
        quality_dict_rec = self.quality_model(img_ori, img_rec)
        quality_dict_pred = self.quality_model(img_ori, img_pred_rec)
        
        # perform entropy coding
        byte_stream, byte_num, information = encoding_masked_torchac(
            z_indices.to(dtype=torch.int16),
            pmf_container.to(dtype=torch.float32),
            mask
        )
        enc_bpp = byte_num * 8 / pix_num     # calculate bpp after entropy coding
        entropy_bpp = information / pix_num
        total_compressed_bpp = enc_bpp + mask_compressed_bpp

        # log encoding results
        self.log("test/enc_bpp", enc_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/entropy_bpp", entropy_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/mask_bpp", mask_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/mask_compressed_bpp", mask_compressed_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/total_compressed_bpp", total_compressed_bpp, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log predicted quality results
        self.log("test/pred_psnr", quality_dict_pred['psnr'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_msssim", quality_dict_pred['msssim'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_lpips", quality_dict_pred['lpips'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/pred_dists", quality_dict_pred['dists'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log recon quality results
        self.log("test/rec_psnr", quality_dict_rec['psnr'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_msssim", quality_dict_rec['msssim'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_lpips", quality_dict_rec['lpips'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/rec_dists", quality_dict_rec['dists'], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # log images
        mask_large = mask.repeat(3, 1, 1).view(B, 3, H_size, W_size)
        mask_large = F.interpolate(mask_large.float(), size=[H, W], mode='nearest')
        log_imgs = torch.cat([img_ori, img_rec, mask_large, img_pred_rec], dim=0)
        log_imgs = make_grid(log_imgs, padding=10)
        self.logger.experiment.add_image("test/imgs", log_imgs, batch_idx)
        
        # result dict overall
        result_dict = {
            # log encoding results
            "enc_bpp": enc_bpp,
            "entropy_bpp": entropy_bpp,
            "mask_bpp": mask_bpp,
            "mask_compressed_bpp": mask_compressed_bpp,
            "total_compressed_bpp": total_compressed_bpp,
            
            # log predicted quality results
            "PSNR_pred": quality_dict_pred['psnr'],
            "MS-SSIM_pred": quality_dict_pred['msssim'],
            "LPIPS_pred": quality_dict_pred['lpips'],
            "DISTS_pred": quality_dict_pred['dists'],
            
            # log predicted quality results
            "PSNR_rec": quality_dict_rec['psnr'],
            "MS-SSIM_rec": quality_dict_rec['msssim'],
            "LPIPS_rec": quality_dict_rec['lpips'],
            "DISTS_rec": quality_dict_rec['dists'],
        }
        
        # image dict
        img_dict = {
            "original": img_ori,
            "reconstruction": img_rec,
            "predicted": img_pred_rec,
            "mask": mask_large,
            "concated": log_imgs
        }
        
        return {
            'loss': 0.0,
            'result': result_dict,
            'imgs': img_dict
        }
        

class Transformer_stage1_serial_convert(Transformer_stage1_serial):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 trainging_config,
                 test_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 ):
        super().__init__(
                 transformer_config,
                 first_stage_config,
                 trainging_config,
                 test_config,
                 permuter_config,
                 ckpt_path,
                 ignore_keys,
                 first_stage_key,
                 monitor,
        )
        
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        ignore_keys += [
            "transformer.tok_emb",
            "transformer.head",
            "first_stage_model"
        ]
        sd = torch.load(path, map_location="cpu")["state_dict"]
        new_sd = {}
        for k, v in sd.items():
            find = False
            
            # search ignore keys
            for i_k in ignore_keys:
                if k.startswith(i_k):
                    find = True
            
            if not find:
                new_sd[k] = v
                
        self.load_state_dict(new_sd, strict=False)
        print(f"Restored from {path}")
        

    def configure_optimizers(self):
        params_dict = dict(self.transformer.named_parameters())
        target_names = ["tok_emb.weight", "head.weight"]
        
        optimizer = torch.optim.AdamW(
                (params_dict[name] for name in target_names), 
                lr=self.init_learning_rate
            )
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=self.lr_milestone, 
                    gamma=self.lr_milestones_gamma
                ),
                'interval': self.scheduler_interval,
                'frequency': 1,
            }
        }
        return optim_dict