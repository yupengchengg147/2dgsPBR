import os
import sys
import torch
from random import randint
from tqdm import tqdm
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from scene import Scene, GaussianModel
from pbr import CubemapLight, get_brdf_lut


from gaussian_renderer import pbr_render, network_gui, render

from train import prepare_output_and_logger, training_report


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def pbr_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, metallic=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    cubemap = CubemapLight(base_res=256).cuda()
    cubemap.train()

    param_groups = [
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.brdf_mlp_lr_init},
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=opt.brdf_mlp_lr_init,
                                        lr_final=opt.brdf_mlp_lr_final,
                                        lr_delay_mult=opt.brdf_mlp_lr_delay_mult,
                                        max_steps=opt.brdf_mlp_lr_max_steps)
    
    #TODO: 
    #     3. object mask for normal loss and distortion loss
    #     4.verify get_normal 
    #     5. verify distort
    #    6. gamma correction

    brdf_lut = get_brdf_lut().cuda()

    if checkpoint:
        (model_params, light_params, lightopt_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        cubemap.load_state_dict(light_params)
        light_optimizer.load_state_dict(lightopt_params)
        print("Restored from checkpoint at iteration", first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    
    # in warmup iterations, set envmap gradient false, set pbr related para gradient false
    cubemap.base.requires_grad = False
    gaussians.set_requires_grad("albedo", False)
    gaussians.set_requires_grad("roughness", False)
    gaussians.set_requires_grad("metallic", False)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        #dynamic lr
        gaussians.update_learning_rate(iteration)
        

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if iteration < opt.warmup_iterations:
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            normal_from_d = render_pkg['surf_normal']
        elif iteration == opt.warmup_iterations:
            cubemap.base.requires_grad = True
        
            gaussians.set_requires_grad("features_dc", False)
            gaussians.set_requires_grad("features_rest", False)

            gaussians._albedo.data = gaussians._features_dc.data.clone().squeeze()
            gaussians.set_requires_grad("albedo", True)
            gaussians.set_requires_grad("roughness", True)
            
            # if not metallic:
            #     gaussians._metallic = None
            # else:
            #     gaussians.set_requires_grad("metallic", True)
            gaussians.set_requires_grad("metallic", True)
            continue
        else:
            for param_group in light_optimizer.param_groups:
                if param_group["name"] == "cubemap":
                    lr = brdf_mlp_scheduler_args(iteration - opt.warmup_iterations)
                    param_group['lr'] = lr
            cubemap.build_mips()
            render_pkg = pbr_render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            light=cubemap,
            pipe=pipe,
            bg_color=background,
            brdf_lut=brdf_lut,
            speed=True)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            alpha, rend_normal, dist, surf_depth, normal_from_d = render_pkg["rend_alpha"], render_pkg["rend_normal"], render_pkg["rend_dist"], render_pkg["surf_depth"], render_pkg["surf_normal"]

            # diffuse_rgb, specular_rgb, albedo, roughness, metallic = render_pkg["diffuse_rgb"], render_pkg["specular_rgb"], render_pkg["albedo"], render_pkg["roughness"], render_pkg["metallic"]
            # mask = (render_pkg["rend_alpha"] != 0).all(0)[None,:,:]
            # image = torch.where(mask, render_pkg["render"], background[:,None,None])

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # normal_error = (1 - (rend_normal[mask.repeat(3,1,1)] * normal_from_d[mask.repeat(3,1,1)]).sum(dim=0))[None]
        normal_error = (1 - (rend_normal * normal_from_d).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        # dist_loss = lambda_dist * (dist[mask]).mean()
        dist_loss = lambda_dist * (dist).mean()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
            
            if iteration < opt.warmup_iterations:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            else:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, pbr_render, 
                                (cubemap, pipe, background, brdf_lut)
                                )
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration > opt.warmup_iterations: 
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0,max=1.0)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), cubemap.state_dict(),light_optimizer.state_dict(),iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        



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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--metallic', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # dataset = lp.extract(args)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pbr_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.metallic)

    # All done
    print("\nTraining complete.")
    # warmup_iterations = 7001
    # lambda_dist 100 -bounded 1000 -unbounded