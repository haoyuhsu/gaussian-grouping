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

import copy

from edit_object_removal import points_inside_convex_hull

import trimesh
import open3d as o3d


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


def extract_geometry_and_render(dataset : ModelParams, iteration : int, pipeline : PipelineParams, custom_traj_name : str, selected_obj_ids : int, removal_thresh : float, OBJECT_NAME : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get the info of custom camera trajectory
        custom_traj_folder = os.path.join(dataset.source_path)
        with open(os.path.join(custom_traj_folder, custom_traj_name + '.json'), 'r') as f:
            custom_traj = json.load(f)

        # get camera poses and intrinsics
        fx, fy, cx, cy = custom_traj["fl_x"], custom_traj["fl_y"], custom_traj["cx"], custom_traj["cy"]
        w, h = custom_traj["w"], custom_traj["h"]
        c2w_dict = {}
        for frame in custom_traj["frames"]:
            c2w_dict[frame["filename"]] = np.array(frame["transform_matrix"])

        resolution = 4   # TODO: support different resolutions
        if resolution in [1, 2, 4, 8]:
            w, h = int(w / resolution), int(h / resolution)
            fx, fy = fx / resolution, fy / resolution
            cx, cy = cx / resolution, cy / resolution

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        c2w_dict = dict(sorted(c2w_dict.items()))
        
        # setting up tracking folder path
        object_name = OBJECT_NAME
        deva_output_path = os.path.join(dataset.source_path, "custom_camera_path", custom_traj_name, 'track_with_deva')
        tracking_dir = os.path.join(deva_output_path, object_name)

        # get the tracking binary masks for the object
        obj_masks = {}
        if os.path.exists(tracking_dir):
            instance_id = [f for f in os.listdir(tracking_dir) if f.isdigit()]   # get the folder named by instance ID
            obj_folder = os.path.join(tracking_dir, instance_id[0])              # only one object instance for demo (TODO: support multiple instances)
            for file in os.listdir(obj_folder):
                if file.endswith(".png"):
                    mask = cv2.imread(os.path.join(obj_folder, file), cv2.IMREAD_GRAYSCALE)
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)    # erode the mask to remove the noisy boundary
                    obj_masks[file] = mask
        else:
            raise FileNotFoundError(f"Tracking masks not found at {tracking_dir}")
        
        # sort the obj_masks by filename (all frames that consist the object instance)
        obj_masks = dict(sorted(obj_masks.items()))

        print("Number of object masks: ", len(obj_masks))
        print("Number of camera poses: ", len(c2w_dict))

        scene_mesh_path = '/home/shenlong/Documents/maxhsu/3D-Scene-Editing/output/garden_norm/sdf-mesh/xtlas_simplify/mesh.obj'
        scene_mesh = trimesh.load_mesh(scene_mesh_path)

        print("Number of vertices:", len(scene_mesh.vertices))
        print("Number of faces:", len(scene_mesh.faces))

        # compute closest triangle of each gaussians
        gaussians_xyz = gaussians._xyz.cpu().numpy()
        # closest, distances, triangle_ids = trimesh.proximity.closest_point(scene_mesh, gaussians_xyz)  # slow and will freeze the program
        scene_o3d = o3d.t.geometry.RaycastingScene()
        scene_o3d.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh.as_open3d))
        ret_dict = scene_o3d.compute_closest_points(o3d.core.Tensor.from_numpy(gaussians_xyz.astype(np.float32)))
        triangle_ids = ret_dict['primitive_ids'].cpu().numpy()

        TRIANGLES_VIEW_COUNTER = torch.zeros(len(scene_mesh.faces), dtype=torch.int32, device="cuda")
        for filename, mask in tqdm(obj_masks.items(), desc="Unprojecting..."):
            
            idx = int(filename.split('/')[-1].split('.')[0])

            mask = torch.tensor(mask, dtype=torch.bool, device="cuda")

            c2w = c2w_dict[filename]
            c2w = torch.FloatTensor(c2w).to("cuda")
            directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
            rays_d = directions @ c2w[:3, :3].T
            rays_o = c2w[:3, 3].expand_as(rays_d)

            rays_d = rays_d[mask].reshape(-1, 3).cpu().numpy()
            rays_o = rays_o[mask].reshape(-1, 3).cpu().numpy()

            index_tri = scene_mesh.ray.intersects_first(
                ray_origins=rays_o,
                ray_directions=rays_d
            )

            index_tri = torch.tensor(index_tri, dtype=torch.int32, device="cuda")
            TRIANGLES_VIEW_COUNTER[index_tri] += 1

        N_VIEWS = len(obj_masks.keys())

        torch.cuda.empty_cache()
        
        total_missed_pixels_list = []
        RATIO_LIST = [0.25, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.7]    # TODO: might use binary search to find the optimal ratio
        for RATIO in RATIO_LIST:
            print("===== RATIO: ", RATIO, "=====")
            MINIMUM_VIEWS = int(N_VIEWS * RATIO)  # minimum number of views to consider a gaussian as close to the object

            mask_triangles = TRIANGLES_VIEW_COUNTER >= MINIMUM_VIEWS
            print("Number of triangles that are close to the object: ", mask_triangles.sum())

            mask_triangles = mask_triangles.cpu().numpy()
            masked_mesh = scene_mesh.submesh([mask_triangles], append=True)
            convex_hull = masked_mesh.convex_hull
            original_tri_centroids = scene_mesh.triangles_center
            inside_hull = convex_hull.contains(original_tri_centroids)
            mask_triangles = np.logical_or(mask_triangles, inside_hull)
            mask_triangles_idx = np.where(mask_triangles)[0]

            # keep the gaussians with closest triangle index in mask_triangles_idx
            ### mask_trianlges_idx: the index of triangles that are close to the object
            ### triangle_ids: the index of the closest triangle of each gaussian
            mask3d = np.isin(triangle_ids, mask_triangles_idx)
            mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")
            print("Number of gaussians that are close to the object: ", mask3d.sum())

            # crop gaussians
            object_gaussians = copy.deepcopy(gaussians)
            object_gaussians._xyz = gaussians._xyz[mask3d]
            object_gaussians._features_dc = gaussians._features_dc[mask3d]
            object_gaussians._features_rest = gaussians._features_rest[mask3d]
            object_gaussians._scaling = gaussians._scaling[mask3d]
            object_gaussians._rotation = gaussians._rotation[mask3d]
            object_gaussians._opacity = gaussians._opacity[mask3d]
            object_gaussians._objects_dc = gaussians._objects_dc[mask3d]

            total_missed_pixels = 0

            # render the cropped gaussians to compute the object mask
            # TODO: c2w_dict might not have the same length as obj_masks.keys()
            for idx, c2w in enumerate(tqdm(c2w_dict.values(), desc="Rendering progress")):
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
                FovY = focal2fov(fy, h)
                FovX = focal2fov(fx, w)
                image = np.zeros((h, w, 4), dtype=np.uint8)
                cam_info = CameraInfo(uid=1, R=R, T=T, FovY=FovY, FovX=FovX, image=Image.fromarray(image),
                                    image_path=None, image_name='{0:05d}'.format(idx), width=w, height=h,
                                    objects=None)
                view = customLoadCam(-1, idx, cam_info)
                results = render(view, object_gaussians, pipeline, background)
                object_alpha_image = results["render"].permute(1, 2, 0)[:, :, 3].cpu().numpy()
                object_mask_rendered = object_alpha_image >= 0.8
                object_mask_tracking = np.array(obj_masks[list(obj_masks.keys())[idx]]) == 255
                # compute the number of pixels mismatched by XOR operation
                xor_mask = np.logical_xor(object_mask_rendered, object_mask_tracking)
                # save the above masks for debugging
                # if idx == 0:
                #     tmp_folder = './test_masks/RATIO={}'.format(RATIO)
                #     os.makedirs(tmp_folder, exist_ok=True)
                #     cv2.imwrite(os.path.join(tmp_folder, f"object_mask_rendered_{idx}.png"), object_mask_rendered.astype(np.uint8) * 255)
                #     cv2.imwrite(os.path.join(tmp_folder, f"object_mask_tracking_{idx}.png"), object_mask_tracking.astype(np.uint8) * 255)
                #     cv2.imwrite(os.path.join(tmp_folder, f"xor_mask_{idx}.png"), xor_mask.astype(np.uint8) * 255)
                xor_pixels = np.count_nonzero(xor_mask)
                total_missed_pixels += xor_pixels
            
            print("Total missed pixels: ", total_missed_pixels)
            total_missed_pixels_list.append(total_missed_pixels)

        for ratio, missed_pixels in zip(RATIO_LIST, total_missed_pixels_list):
            print("RATIO: ", ratio, "MISSED PIXELS: ", missed_pixels)            

        # select the ratio with minimum missed pixels
        best_ratio = RATIO_LIST[np.argmin(total_missed_pixels_list)]
        print("===== BEST RATIO: ", best_ratio, "=====")

        MINIMUM_VIEWS = int(N_VIEWS * best_ratio)  # minimum number of views to consider a gaussian as close to the object
        mask_triangles = TRIANGLES_VIEW_COUNTER >= MINIMUM_VIEWS
        # mask_triangles_idx = mask_triangles.nonzero().flatten().cpu().numpy()
        mask_triangles = mask_triangles.cpu().numpy()
        masked_mesh = scene_mesh.submesh([mask_triangles], append=True)
        convex_hull = masked_mesh.convex_hull
        original_tri_centroids = scene_mesh.triangles_center
        inside_hull = convex_hull.contains(original_tri_centroids)
        mask_triangles = np.logical_or(mask_triangles, inside_hull)
        mask_triangles_idx = np.where(mask_triangles)[0]
        mask3d = np.isin(triangle_ids, mask_triangles_idx)
        mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")

        save_dir = os.path.join(dataset.model_path, "object_instance", '_'.join(object_name.split(' ')))
        os.makedirs(save_dir, exist_ok=True)

        # save the convex hull meshes (TODO: might be blurry)
        # convex_hull_mesh_dir = os.path.join(save_dir, "convex_hull_mesh")
        # os.makedirs(convex_hull_mesh_dir, exist_ok=True)
        # convex_hull_mesh = trimesh.Trimesh(vertices=convex_hull.vertices, faces=convex_hull.faces)
        # convex_hull_mesh.export(os.path.join(convex_hull_mesh_dir, "object_convex_hull_mesh.obj"))

        # save object instance mesh
        object_mesh_save_dir = os.path.join(save_dir, "object_mesh")
        os.makedirs(object_mesh_save_dir, exist_ok=True)
        object_instance_mesh = scene_mesh.submesh([mask_triangles], append=True)
        object_instance_mesh.export(os.path.join(object_mesh_save_dir, "object_mesh.obj"))

        # save the remaining scene mesh
        scene_mesh_save_dir = os.path.join(save_dir, "removal_mesh")
        os.makedirs(scene_mesh_save_dir, exist_ok=True)
        non_mask_triangles = ~mask_triangles
        scene_mesh_remaining = scene_mesh.submesh([non_mask_triangles], append=True)
        scene_mesh_remaining.export(os.path.join(scene_mesh_save_dir, "removal_mesh.obj"))

        # save object instance gaussians for further usage
        object_gaussians = copy.deepcopy(gaussians)
        object_gaussians._xyz = gaussians._xyz[mask3d]
        object_gaussians._features_dc = gaussians._features_dc[mask3d]
        object_gaussians._features_rest = gaussians._features_rest[mask3d]
        object_gaussians._scaling = gaussians._scaling[mask3d]
        object_gaussians._rotation = gaussians._rotation[mask3d]
        object_gaussians._opacity = gaussians._opacity[mask3d]
        object_gaussians._objects_dc = gaussians._objects_dc[mask3d]
        object_gaussians.save_ply(os.path.join(save_dir, "object_gaussians.ply"))

        # save the remaining gaussians for further usage
        non_mask3d = ~mask3d
        object_removal_gaussians = copy.deepcopy(gaussians)
        object_removal_gaussians._xyz = gaussians._xyz[non_mask3d]
        object_removal_gaussians._features_dc = gaussians._features_dc[non_mask3d]
        object_removal_gaussians._features_rest = gaussians._features_rest[non_mask3d]
        object_removal_gaussians._scaling = gaussians._scaling[non_mask3d]
        object_removal_gaussians._rotation = gaussians._rotation[non_mask3d]
        object_removal_gaussians._opacity = gaussians._opacity[non_mask3d]
        object_removal_gaussians._objects_dc = gaussians._objects_dc[non_mask3d]
        object_removal_gaussians.save_ply(os.path.join(save_dir, "removal_gaussians.ply"))

        # recomposition of the scene (ex: plus 0.25 in x-axis for object gaussians)
        recomposed_gaussians = copy.deepcopy(gaussians)
        recomposed_gaussians._xyz[mask3d] = gaussians._xyz[mask3d] + torch.tensor([0.25, 0, 0], dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "custom_camera_path", custom_traj_name, "images_removal_object_id={}".format(selected_obj_ids))
        makedirs(render_path, exist_ok=True)
        obj_render_path = os.path.join(dataset.model_path, "custom_camera_path", custom_traj_name, "images_object_id={}".format(selected_obj_ids))
        makedirs(obj_render_path, exist_ok=True)
        recomposed_render_path = os.path.join(dataset.model_path, "custom_camera_path", custom_traj_name, "images_recomposited_object_id={}".format(selected_obj_ids))
        makedirs(recomposed_render_path, exist_ok=True)

        # render the object with custom camera trajectory
        for idx, c2w in enumerate(tqdm(c2w_dict.values(), desc="Rendering progress")):
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            FovY = focal2fov(fy, h)
            FovX = focal2fov(fx, w)
            image = np.zeros((h, w, 4), dtype=np.uint8)
            cam_info = CameraInfo(uid=1, R=R, T=T, FovY=FovY, FovX=FovX, image=Image.fromarray(image),
                                image_path=None, image_name='{0:05d}'.format(idx), width=w, height=h,
                                objects=None)
            view = customLoadCam(-1, idx, cam_info)
            # view = Camera(
            #     colmap_id=1, R=R, T=T, 
            #     FoVx=FovX, FoVy=FovY, image=torch.zeros(4, h, w), gt_alpha_mask=None, 
            #     image_name='{0:05d}'.format(idx), uid=idx)

            results = render(view, object_removal_gaussians, pipeline, background)
            torchvision.utils.save_image(results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            results = render(view, object_gaussians, pipeline, background)
            torchvision.utils.save_image(results["render"], os.path.join(obj_render_path, '{0:05d}'.format(idx) + ".png"))

            results = render(view, recomposed_gaussians, pipeline, background)
            torchvision.utils.save_image(results["render"], os.path.join(recomposed_render_path, '{0:05d}'.format(idx) + ".png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_iteration", type=int, default=0, help="Load a specific iteration from a checkpoint")
    parser.add_argument("--selected_obj_ids", type=int, default=-1, help="Object ids to be removed")
    parser.add_argument("--removal_thresh", type=float, default=0.5, help="Threshold for object removal")
    parser.add_argument("--OBJECT_NAME", type=str, default="object", help="Object name to be removed")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    extract_geometry_and_render(model.extract(args), args.iteration, pipeline.extract(args), args.custom_traj_name, args.selected_obj_ids, args.removal_thresh, args.OBJECT_NAME)