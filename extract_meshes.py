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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import json
import numpy as np
import cv2
import math
from scene import Scene
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
from kornia import create_meshgrid
# from tracking.demo_with_text import run_deva

# from render_custom_camera_path import customLoadCam
from utils.general_utils import PILtoTorch
from PIL import Image
from render import feature_to_rgb, id2rgb, visualize_obj


def customLoadCam(resolution, id, cam_info, resolution_scale=1.0, data_device="cuda"):
    orig_w, orig_h = cam_info.image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                # print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                #     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=data_device)


def depth2img(depth, scale=16):
    depth = depth / scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_img


# @torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True, anti_aliasing_factor=1.0):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    if anti_aliasing_factor > 1.0:
        H = int(H * anti_aliasing_factor) 
        W = int(W * anti_aliasing_factor) 
        K *= anti_aliasing_factor
        K[2, 2] = 1
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


# @torch.cuda.amp.autocast(dtype=torch.float32)
# def get_rays(directions, c2w):
#     """
#     Get ray origin and directions in world coordinate for all pixels in one image.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems

#     Inputs:
#         directions: (N, 3) ray directions in camera coordinate
#         c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

#     Outputs:
#         rays_o: (N, 3), the origin of the rays in world coordinate
#         rays_d: (N, 3), the direction of the rays in world coordinate
#     """
#     if c2w.ndim==2:
#         # Rotate ray directions from camera coordinate to the world coordinate
#         rays_d = directions @ c2w[:, :3].T
#     else:
#         rays_d = rearrange(directions, 'n c -> n 1 c') @ \
#                  rearrange(c2w[..., :3], 'n a b -> n b a')
#         rays_d = rearrange(rays_d, 'n 1 c -> n c')
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[..., 3].expand_as(rays_d)

#     return rays_o, rays_d 


def extract_geometry(dataset : ModelParams, iteration : int, pipeline : PipelineParams, custom_traj_name : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = scene.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get the info of custom camera trajectory
        custom_traj_folder = os.path.join(dataset.source_path)
        with open(os.path.join(custom_traj_folder, custom_traj_name + '.json'), 'r') as f:
            custom_traj = json.load(f)

        render_path = os.path.join(dataset.model_path, "custom_camera_path", custom_traj_name, "images")
        makedirs(render_path, exist_ok=True)
        obj_render_path = os.path.join(dataset.model_path, "custom_camera_path", custom_traj_name, "object_images")
        makedirs(obj_render_path, exist_ok=True)

        # get camera poses and intrinsics
        fx, fy, cx, cy = custom_traj["fl_x"], custom_traj["fl_y"], custom_traj["cx"], custom_traj["cy"]
        w, h = custom_traj["w"], custom_traj["h"]
        c2w_dict = {}
        for frame in custom_traj["frames"]:
            c2w_dict[frame["filename"]] = np.array(frame["transform_matrix"])

        c2w_dict = dict(sorted(c2w_dict.items()))

        # render images
        # for idx, c2w in enumerate(tqdm(c2w_dict.values(), desc="Rendering progress")):
        #     w2c = np.linalg.inv(c2w)
        #     R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        #     T = w2c[:3, 3]
        #     FovY = focal2fov(fy, h)
        #     FovX = focal2fov(fx, w)
        #     image = np.zeros((h, w, 4), dtype=np.uint8)
        #     depth = np.zeros((h, w))
        #     cam_info = CameraInfo(uid=1, R=R, T=T, FovY=FovY, FovX=FovX, image=Image.fromarray(image),
        #                       image_path=None, image_name='{0:05d}'.format(idx), width=w, height=h,
        #                       objects=None)
        #     view = customLoadCam(-1, idx, cam_info)
        #     # view = Camera(
        #     #     colmap_id=1, R=R, T=T, 
        #     #     FoVx=FovX, FoVy=FovY, image=torch.zeros(4, h, w), gt_alpha_mask=None, 
        #     #     image_name='{0:05d}'.format(idx), uid=idx)
        #     results = render(view, gaussians, pipeline, background)

        #     # save rendered images
        #     rgba_img = results["render"]
        #     torchvision.utils.save_image(rgba_img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        #     # save instance segmentation masks
        #     rendering_obj = results["render_object"]
        #     logits = classifier(rendering_obj)
        #     pred_obj = torch.argmax(logits,dim=0)
        #     pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        #     cv2.imwrite(os.path.join(obj_render_path, '{0:05d}'.format(idx) + ".png"), cv2.cvtColor(pred_obj_mask, cv2.COLOR_RGB2BGR))

            # save depth images
            # depth_raw = result["depth"].permute(1, 2, 0).cpu().numpy()
            # depth = depth2img(depth_raw, scale=16)
            # cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))

        # print("Rendering finished")

        gaussians_features = gaussians._objects_dc
        gaussians_features = gaussians_features.permute(2, 0, 1).unsqueeze(0) # (n_gaussians, 1, num_objects) -> (1, num_objects, n_gaussians, 1)
        labels = classifier(gaussians_features).squeeze(0).squeeze(-1).permute(1, 0)    # (1, num_classes, n_gaussians, 1) -> (n_gaussians, num_classes)
        pred_obj = torch.argmax(labels, dim=1)
        # count number of objects per class
        print(torch.count_nonzero(pred_obj))
        # print(torch.count_nonzero(labels, dim=1))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--custom_traj_name", default=None, type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    extract_geometry(model.extract(args), args.iteration, pipeline.extract(args), args.custom_traj_name)