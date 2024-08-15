import os
import sys
import torch
import torch.nn.functional as F

from random import randint

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from scene import Scene, GaussianModel
from pbr import CubemapLight, get_brdf_lut


from gaussian_renderer import pbr_render, network_gui, render, pbr_render_gshader, pbr_render_deffered

import matplotlib.pyplot as plt
import numpy as np


def show_rgb_image(tensor):
    # 将张量从 CUDA 转移到 CPU，并将其转换为 NumPy 数组
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # 变更维度顺序从 [C, H, W] 到 [H, W, C]
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def show_single_channel_image(tensor):
    # 将张量从 CUDA 转移到 CPU，并将其转换为 NumPy 数组
    img = tensor.detach().cpu().numpy().squeeze(0)  # 移除通道维度 [1, H, W] -> [H, W]
    plt.imshow(img, cmap='gray')  # 使用灰度色图
    plt.axis('off')  # 不显示坐标轴
    plt.show()


import sys
from argparse import ArgumentParser

# 模拟命令行参数
sys.argv = ['pbr_train.py', '--source_path', '../data/ref_synthetic/toaster/', '--model_path', './all_test/test_toaster/']

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--start_checkpoint", type=str, default = None)
# parser.add_argument('--source_path', type=str, required=True)
# parser.add_argument('--model_path', type=str, required=True)
# --white_background 
# parser.add_argument('--white_background', action='store_true', default=False)
# # --eval 
# parser.add_argument('--eval', action='store_true', default=False)
# # --resolution
# parser.add_argument('--resolution', type=int, default=1)

# 解析参数
args = parser.parse_args()

pipe = pp.extract(args)
opt = op.extract(args)
dataset = lp.extract(args)

chkp_path = "/root/autodl-tmp/2dgsPBR/all_test/test_toaster/chkpnt30000.pth"
lp._source_path = "../data/ref_synthetic/toaster/"
lp._model_path = "../all_test/test_toaster/"

(model_params, light_params, _, first_iter) = torch.load(chkp_path)

gaussians = GaussianModel(lp.sh_degree)
scene = Scene(dataset, gaussians, shuffle=False)
gaussians.restore(model_params)

cubemap = CubemapLight(base_res=256).cuda()
cubemap.load_state_dict(light_params)
cubemap.eval()

print("Restored from checkpoint at iteration", first_iter)

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

viewpoint_stack = scene.getTrainCameras().copy()
viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
brdf_lut = get_brdf_lut().cuda()
cubemap.build_mips()


canonical_rays = scene.get_canonical_rays()
canonical_rays.shape, canonical_rays
H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
view_dirs = -(
    (
        F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
        * c2w[None, :3, :3]
    )  # [HW, 3, 3]
    .sum(dim=-1)
    .reshape(H, W, 3)
)
# canonical_rays是由相机中心指向外部的，所以取负号是wo的方向

render_pkg = pbr_render_deffered(viewpoint_camera=viewpoint_cam,
                                             pc=gaussians,
                                             light=cubemap,
                                             pipe=pipe,
                                             bg_color=background,
                                             view_dirs=view_dirs,
                                             brdf_lut=brdf_lut,
                                             speed=False)

image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
alpha, rend_normal, dist, surf_depth, normal_from_d = render_pkg["rend_alpha"], render_pkg["rend_normal"], render_pkg["rend_dist"], render_pkg["surf_depth"], render_pkg["surf_normal"]
kd_map, ks_map, kr_map, cr_map = render_pkg["kd_map"], render_pkg["ks_map"], render_pkg["kr_map"], render_pkg["cr_map"]
diffuse_color, specular_color = render_pkg["diffuse_color"], render_pkg["specular_color"]  
spec0, spec1 = render_pkg["spec0"], render_pkg["spec1"]