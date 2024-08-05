import torch
from random import randint
import os
import sys

from gaussian_renderer import pbr_render
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.general_utils import safe_state
from pbr import CubemapLight, get_brdf_lut

from train import prepare_output_and_logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    dataset.white_background = True
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cubemap = CubemapLight(base_res=256).cuda()
    cubemap.build_mips()

    brdf_lut = get_brdf_lut().cuda()

    viewpoint_stack = None
    if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    rets = pbr_render(
          viewpoint_camera=viewpoint_cam,
          pc=gaussians,
          light=cubemap,
          pipe=pipe,
          bg_color=background,
          brdf_lut=brdf_lut)

    # render_path = 

