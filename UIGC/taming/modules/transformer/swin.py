import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange 
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from taming.modules.vqvae.quantize import VectorQuantizer2


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
        Windows Multi-head Self-Attention (W-MSA)
    """

    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, is_SW: bool):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.is_SW=is_SW
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

        # Relative Position Calculation
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]), dtype=torch.long, requires_grad=False)
        self.relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.is_SW:
            s = p - shift
            attn_mask[-1, :, :s, :, s:, :] = True
            attn_mask[-1, :, s:, :, :s, :] = True
            attn_mask[:, -1, :, :s, :, s:] = True
            attn_mask[:, -1, :, s:, :, :s] = True
            attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')

        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.is_SW: x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(
            self.relative_position_params[:, self.relation[:,:,0], self.relation[:,:,1]], 
            'h p q -> h 1 1 p q')
        if self.is_SW:
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.is_SW: output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output


class SwinTBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, drop_path: float, is_SW: bool):
        """ SwinTransformer Block, input must be [b h w c]
        """
        super(SwinTBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_SW = is_SW
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.is_SW)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        self.adj = nn.Linear(input_dim, output_dim) \
            if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        """ input must be [b h w c]
        """
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = self.adj(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x
    

class DualSwinTBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dim: int, 
                 window_size: int, drop_path: float = 0.0, re_arrange: bool = False) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            SwinTBlock(input_dim, input_dim, head_dim, window_size, drop_path, is_SW=False),
            SwinTBlock(input_dim, output_dim, head_dim, window_size, drop_path, is_SW=True),
        )
        self.rearrange_0 = Rearrange('b c h w -> b h w c') if re_arrange else nn.Identity()
        self.rearrange_1 = Rearrange('b h w c -> b c h w') if re_arrange else nn.Identity()
        
    def forward(self, x):
        x = self.rearrange_0(x)
        x = self.blocks(x)
        x = self.rearrange_1(x)
        return x
    
class SwinPrior(nn.Module):
    def __init__(self,
                 vocab_size: int=1024,
                 block_size: int=512,
                #  n_layers: list=[2, 3, 3],
                 head_dim: int=16,
                 n_embd: int=512
                ) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        # self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_embd = n_embd
        
        # embedding
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # stage 0
        self.stage_0 = nn.Parameter(torch.ones((1, vocab_size, 1, 1)), requires_grad=True)
        
        # stage 1
        self.stage_1 = nn.Sequential(
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        # stage 2
        self.stage_2 = nn.Sequential(
            nn.Conv2d(n_embd * 2, 768, kernel_size=1),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        # stage 3
        self.stage_3 = nn.Sequential(
            nn.Conv2d(n_embd * 3, 1024, kernel_size=1),
            DualSwinTBlock(1024, 768, head_dim, 4, re_arrange=True),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        
    def forward(self, x):
        B, H, W = x.shape
        device = x.device
        logits = torch.zeros((B, self.vocab_size, H, W), device=device)
        
        # stage 0
        logits_0 = self.stage_0
        logits[:, :, 0::2, 0::2] = logits_0
        x_0 = x[:, 0::2, 0::2]
        emb = [self.tok_emb(x_0).permute(0, 3, 1, 2)]
        
        # stage 1
        logits_1 = self.stage_1(torch.cat(emb, dim=1))
        logits[:, :, 1::2, 1::2] = logits_1
        x_1 = x[:, 1::2, 1::2]
        emb.append(self.tok_emb(x_1).permute(0, 3, 1, 2))
        
        # stage 2
        logits_2 = self.stage_2(torch.cat(emb, dim=1))
        logits[:, :, 0::2, 1::2] = logits_2
        x_2 = x[:, 0::2, 1::2]
        emb.append(self.tok_emb(x_2).permute(0, 3, 1, 2))
        
        # stage 3
        logits_3 = self.stage_3(torch.cat(emb, dim=1))
        logits[:, :, 1::2, 0::2] = logits_3
        
        return logits
    
    
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

    
class JointHyperPrior(nn.Module):
    def __init__(self,
                 vocab_size: int=1024,
                 block_size: int=512,
                #  n_layers: list=[2, 3, 3],
                 head_dim: int=16,
                 n_embd: int=512
                ) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        # self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_embd = n_embd
        
        # hyper encoder
        self.h_enc = nn.Sequential(
            DualSwinTBlock(n_embd, n_embd, head_dim, 8, re_arrange=True),
            nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=2),
            DualSwinTBlock(n_embd, n_embd, head_dim, 4, re_arrange=True),
            nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=2),
        )
        
        # hyper quantizer
        self.h_q = VectorQuantizer2(256, n_embd, beta=0.25, sane_index_shape=True)
        
        # hyper decoder
        self.h_dec = nn.Sequential(
            nn.ConvTranspose2d(n_embd, n_embd, kernel_size=3, padding=1, stride=2, output_padding=1),
            DualSwinTBlock(n_embd, n_embd, head_dim, 4, re_arrange=True),
            DualSwinTBlock(n_embd, n_embd, head_dim, 4, re_arrange=True),
        )
        
        # stage 0
        self.stage_0 = nn.Sequential(
            DualSwinTBlock(n_embd, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        # stage 1
        self.stage_1_spatial_merge = nn.Sequential(
            nn.Identity(),
        )
        self.stage_1_hyper_merge = nn.Sequential(
            nn.Conv2d(n_embd * 2, 768, kernel_size=1),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
        )
        self.stage_1 = nn.Sequential(
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        # stage 2
        self.stage_2_spatial_merge = nn.Sequential(
            nn.Conv2d(n_embd * 2, 768, kernel_size=1),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
        )
        self.stage_2_hyper_merge = nn.Sequential(
            nn.Conv2d(n_embd * 2, 768, kernel_size=1),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
        )
        self.stage_2 = nn.Sequential(
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
        # stage 3
        self.stage_3_spatial_merge = nn.Sequential(
            nn.Conv2d(n_embd * 3, 1024, kernel_size=1),
            DualSwinTBlock(1024, 768, head_dim, 4, re_arrange=True),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
        )
        self.stage_3_hyper_merge = nn.Sequential(
            nn.Conv2d(n_embd * 2, 768, kernel_size=1),
            DualSwinTBlock(768, 512, head_dim, 4, re_arrange=True),
        )
        self.stage_3 = nn.Sequential(
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            DualSwinTBlock(512, 512, head_dim, 4, re_arrange=True),
            nn.Conv2d(512, 768, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(768, vocab_size, kernel_size=1)
        )
        
    def forward(self, x):
        B, H, W = x.shape
        device = x.device
        logits = torch.zeros((B, self.vocab_size, H, W), device=device)
        
        x_emb = self.tok_emb(x).permute(0, 3, 1, 2)
        
        # hyper prior
        z = self.h_enc(x_emb)
        z_hat, h_vq_loss, h_vq_info = self.h_q(z)
        z_param = self.h_dec(z_hat)
        
        # stage 0
        logits_0 = self.stage_0(z_param)
        logits[:, :, 0::2, 0::2] = logits_0
        emb_list = [x_emb[:, 0::2, 0::2]]
        
        # stage 1
        logits_1 = self.stage_1_spatial_merge(torch.cat(emb_list, dim=1))
        logits_1 = self.stage_1_hyper_merge(torch.cat([logits_1, z_param], dim=1))
        logits_1 = self.stage_1(logits_1)
        logits[:, :, 1::2, 1::2] = logits_1
        emb_list.append(x_emb[:, 1::2, 1::2])
        
        # stage 2
        logits_2 = self.stage_2_spatial_merge(torch.cat(emb_list, dim=1))
        logits_2 = self.stage_2_hyper_merge(torch.cat([logits_2, z_param], dim=1))
        logits_2 = self.stage_2(logits_2)
        logits[:, :, 0::2, 1::2] = logits_2
        emb_list.append(x_emb[:, 0::2, 1::2])
        
        # stage 3
        logits_3 = self.stage_3_spatial_merge(torch.cat(emb_list, dim=1))
        logits_3 = self.stage_3_hyper_merge(torch.cat([logits_3, z_param], dim=1))
        logits_3 = self.stage_3(logits_3)
        logits[:, :, 1::2, 0::2] = logits_3
        
        return logits