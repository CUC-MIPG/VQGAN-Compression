import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
import zlib
from bitstream import BitStream
from torchvision.models.detection import (
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights
)
from torchvision.models._meta import _COCO_CATEGORIES


# ----------------- mask feature models -----------------

class FCOS(torch.nn.Module):  # segmentation
    def __init__(self, down_factor) -> None:
        super().__init__()
        self.down_factor = down_factor
        self.detection_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT).eval()
        self.pool = nn.AvgPool2d(kernel_size=self.down_factor)

    @torch.no_grad()
    def forward(self, img):
        """return 0.0 is the background, 1.0 is meaningful content
        """
        B, C, H, W = img.shape
        mask = torch.zeros([B, H, W], device=img.device, dtype=torch.float32)
        predictions = self.detection_model(img)
        pred_dict = predictions[0]
        
        # extract box 0
        box_pos = predictions[0]['boxes'][0].to(dtype=torch.int)
        
        # expand region for better quality
        x1, y1, x2, y2 = box_pos[0], box_pos[1], box_pos[2], box_pos[3]
        x1 = x1 - self.down_factor if x1 - self.down_factor >= 0 else x1
        x2 = x2 + self.down_factor
        y1 = y1 - self.down_factor if y1 - self.down_factor >= 0 else y1
        y2 = y2 + self.down_factor
        
        # generate mask
        mask[:, y1:y2, x1:x2] = 1
        mask_down = self.pool(mask)
        
        info = {
            'main_box': box_pos,
            'details': predictions
        }
        return mask, mask_down, info


class Edge_detector(torch.nn.Module):
    def __init__(self, down_factor, is_erode:bool = False) -> None:
        super().__init__()
        from taming.mask_utils.edge_detector import Network as Edge_net
        self.is_erode = is_erode
        self.down_factor = down_factor
        self.model = Edge_net().eval()
        self.down_pool = nn.AvgPool2d(kernel_size=down_factor)
        self.erode_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    
    @torch.no_grad()
    def forward(self, img):
        """return 1.0 is the edge, 0.0 is non-edge
        """
        edge = self.model(torch.flip(img, dims=[1])).clip(0.0, 1.0)
        edge_down = self.down_pool(edge)
        # edge_down = F.interpolate(edge, 
        #                           (edge.shape[2] // self.down_factor, edge.shape[3] // self.down_factor), 
        #                           mode="bilinear")
        edge = edge.squeeze(1)
        edge_down = edge_down.squeeze(1)
        edge_down_binary = edge_down >= torch.mean(edge_down)
        edge_down_binary = edge_down_binary.int()
        
        if self.is_erode:
            edge_down_binary = self.erode_pool(edge_down_binary)
            
        return edge_down_binary, {
            "edge": edge,
            "edge_down": edge_down
        }


class SobelEdge(torch.nn.Module):
    def __init__(self, down_factor) -> None:
        super().__init__()
        self.down_factor = down_factor
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
        self.conv_op = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.conv_op.weight.data = torch.from_numpy(sobel_kernel)
        self.pool = nn.AvgPool2d(kernel_size=self.down_factor)

    @torch.no_grad()
    def forward(self, img):
        """return edge intensity, in float32
        """
        edge_grad = self.conv_op(img).sum(dim=1).abs()
        edge_grad_down = self.pool(edge_grad)
        return edge_grad, edge_grad_down


# ----------- Mask models -------------

class Mask_Model_4step(torch.nn.Module):
    def __init__(self, 
                 mask_type: str,
                 down_mask: bool=False, 
                 preserve_margin: bool=False,
                 down_factor: int=16):
        super().__init__()
        self.dect = FCOS(down_factor).eval()
        self.sobel = SobelEdge(down_factor).eval()
        self.edge = Edge_detector(down_factor).eval()
        
        # mask type
        self.mask_type = mask_type
        
        # down and up sample
        self.down_mask = down_mask
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_weight = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # preserve margin
        self.preserve_margin = preserve_margin
        

    def mask_compress(self, mask: torch.Tensor):
        if len(mask.shape) == 4:
            mask = mask[:, :, :, 0]
        mask = mask.to(dtype=torch.bool)
        mask = mask.flatten().detach().cpu().tolist()
        
        more_bits = len(mask) % 8
        if more_bits:
            append_num = 8 - more_bits
            for i in range(append_num):
                mask.append(True)

        mask_string = BitStream(mask, bool).read(bytes)
        compressed_string = zlib.compress(mask_string)
        return len(mask_string), len(compressed_string)
    

    @torch.no_grad()
    def forward(self, img, z_indices):
        img = img / 2.0 + 0.5
        dect_map, dect_map_down, dect_info = self.dect(img)
        sobel_edge, sobel_edge_down = self.sobel(img)
        edge_map, edge_dict = self.edge(img)
        
        dect_map_down = dect_map_down.int().to(dtype=torch.bool)
        
        if self.mask_type == "seg_edge" or self.mask_type == "seg_edge_more":
            preserve_area = dect_map_down | edge_map  
        elif self.mask_type == "seg" or self.mask_type == "seg_more":
            preserve_area = dect_map_down       
        elif self.mask_type == "edge" or self.mask_type == "edge_more":
            preserve_area = edge_map
        elif self.mask_type == "entire":
            preserve_area = torch.ones_like(z_indices)
        elif self.mask_type == "minium":
            preserve_area = torch.zeros_like(z_indices)
        else:
            raise NotImplementedError("Invalid preserve mask type!")

        if self.down_mask:
            preserve_area_down = self.down(preserve_area.float())
            preserve_area_up = self.up(preserve_area_down.unsqueeze(0)).squeeze(0).int()
        else:
            preserve_area_down = preserve_area.float()
            preserve_area_up = preserve_area.int()
        byte_mask, byte_compressed_mask = self.mask_compress(preserve_area_down)
        
        # generate mask pattern
        mask = torch.zeros_like(z_indices)
        if self.mask_type == "seg_more" or self.mask_type == "edge_more":
            mask[:, ::2, ::2] = 1
        elif self.mask_type.endswith("more"):
            mask[:, ::2, ::2] = 1
        else:
            mask[:, ::2, ::2] = 1
            mask[:, 1::2, 1::2] = 1
        mask = mask | preserve_area_up

        # preserve_margin
        if self.preserve_margin:
            mask[:, 0, :] = 1 
            mask[:, -1, :] = 1
            mask[:, :, 0] = 1
            mask[:, :, -1] = 1

        info_dict = {
            "seg_info": dect_info,
            "edge_info": edge_dict,
            "sobel_info": {"sobel_edge": sobel_edge, "sobel_edge_down": sobel_edge_down},
            "seg_map": dect_map_down,
            "edge_map": edge_map,
            "preserve_area": preserve_area,
            "preserve_area_down": preserve_area_down,
            "preserve_area_up": preserve_area_up,
            "byte_mask": byte_mask,
            "byte_compressed_mask": byte_compressed_mask
        }
    
        return mask, info_dict


class Mask_Model_more_4step(torch.nn.Module):
    def __init__(self, 
                 mask_type: str,
                 down_mask: bool=False, 
                 preserve_margin: bool=False,
                 down_factor: int=16):
        super().__init__()
        self.seg = FCOS(down_factor).eval()
        self.sobel = SobelEdge(down_factor).eval()
        self.edge = Edge_detector(down_factor).eval()
        
        # mask type
        self.mask_type = mask_type
        assert mask_type in ["seg", "seg_more", "edge", "edge_more", "entire", "minium"]
        
        # down and up sample
        self.down_mask = down_mask
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_weight = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # preserve margin
        self.preserve_margin = preserve_margin
        

    def mask_compress(self, mask: torch.Tensor):
        if len(mask.shape) == 4:
            mask = mask[:, :, :, 0]
        mask = mask.to(dtype=torch.bool)
        mask = mask.flatten().detach().cpu().tolist()
        
        more_bits = len(mask) % 8
        if more_bits:
            append_num = 8 - more_bits
            for i in range(append_num):
                mask.append(True)

        mask_string = BitStream(mask, bool).read(bytes)
        compressed_string = zlib.compress(mask_string)
        return len(mask_string), len(compressed_string)
    

    @torch.no_grad()
    def forward(self, img, z_indices):
        img = img / 2.0 + 0.5
        seg_map, seg_dict = self.seg(img)
        sobel_edge, sobel_edge_down = self.sobel(img)
        edge_map, edge_dict = self.edge(img)
        
        if self.mask_type == "seg" or self.mask_type == "seg_more":
            preserve_area = seg_map | edge_map    
        elif self.mask_type == "seg_no_edge" or self.mask_type == "seg_no_edge_more":
            preserve_area = seg_map     
        elif self.mask_type == "edge" or self.mask_type == "edge_more":
            preserve_area = edge_map
        elif self.mask_type == "entire":
            preserve_area = torch.ones_like(z_indices)
        elif self.mask_type == "minium":
            preserve_area = torch.zeros_like(z_indices)
        else:
            raise NotImplementedError("Invalid preserve mask type!")

        if self.down_mask:
            preserve_area_down = self.down(preserve_area.float())
            preserve_area_up = self.up(preserve_area_down.unsqueeze(0)).squeeze(0).int()
        else:
            preserve_area_down = preserve_area.float()
            preserve_area_up = preserve_area.int()
        byte_mask, byte_compressed_mask = self.mask_compress(preserve_area_down)
        
        # generate mask pattern
        mask = torch.zeros_like(z_indices)
        if self.mask_type == "seg_more" or self.mask_type == "edge_more":
            mask[:, ::2, ::2] = 1
        elif self.mask_type.endswith("more"):
            mask[:, ::2, ::2] = 1
        else:
            mask[:, ::2, ::2] = 1
            mask[:, 1::2, 1::2] = 1
        mask = mask | preserve_area_up

        # preserve_margin
        if self.preserve_margin:
            mask[:, 0, :] = 1 
            mask[:, -1, :] = 1
            mask[:, :, 0] = 1
            mask[:, :, -1] = 1

        info_dict = {
            "seg_info": seg_dict,
            "edge_info": edge_dict,
            "sobel_info": {"sobel_edge": sobel_edge, "sobel_edge_down": sobel_edge_down},
            "seg_map": seg_map,
            "edge_map": edge_map,
            "preserve_area": preserve_area,
            "preserve_area_down": preserve_area_down,
            "preserve_area_up": preserve_area_up,
            "byte_mask": byte_mask,
            "byte_compressed_mask": byte_compressed_mask
        }
    
        return mask, info_dict