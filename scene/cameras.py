#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Optional
from PIL import Image

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 normal_path: Optional[str] = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.gt_normal = None
        self.gt_normal_mask = None
        if normal_path is not None:
            try:
                self.gt_normal, self.gt_normal_mask = readGT_normal(normal_path, self.data_device)
            except:
                self.gt_normal, self.gt_normal_mask = None, None


def readGT_normal(normal_path, device):
    img = Image.open(normal_path)
    normal_array = np.array(img)
    normal_array_rgb = normal_array[:, :, :3] #[H,W,3]
    h, w, _ = normal_array_rgb.shape
    variance = np.var(normal_array_rgb, axis=-1)
    threshold = 0.001
    invalid_mask = variance < threshold
    normal_mask = np.ones((h, w), dtype=bool) 
    updated_normal_mask = np.where(invalid_mask, 0, normal_mask) # [H,W]

    gt_normal = torch.tensor(normal_array_rgb, dtype=torch.float32).permute(2, 0, 1).to(device)   # [3,H,W]
    gt_normal = gt_normal/255.0 * 2.0 -1.

    gt_normal_mask = torch.tensor(updated_normal_mask, dtype=torch.bool).unsqueeze(0).to(device) # [1,H,W] 

    #TODO： gt_noraml要rescale到什么范围？ [0,1] or [-1,1]

    return gt_normal, gt_normal_mask

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

