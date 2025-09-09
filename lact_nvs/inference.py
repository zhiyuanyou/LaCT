import argparse
import random
import os

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from data import NVSDataset
from model import LaCTLVSM
from utils import rotate_view_c2w


def get_turntable_cameras_with_zoom_in(
    batch_size=1,
    hfov=50,
    num_views=8,
    w=256,
    h=256,
    min_radius=1.7,
    max_radius=3.0,
    elevation=30,
    up_vector=np.array([0, 0, 1]),
    device=torch.device("cuda"),
):
    '''
    rotate the camera around the object, and change the radius and elevation periodically
    '''
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0) # [num_views, 4]
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.ones_like(azimuths) * (elevation + 15 * np.sin(np.linspace(0, 2*np.pi, num_views)))
    radius = (min_radius + max_radius) / 2.0 + (max_radius - min_radius) / 2.0 * np.sin(np.linspace(0, 2*np.pi, num_views))
    c2ws = []

    for cur_radius, elev, azim in zip(radius, elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = cur_radius * np.sin(elev)
        base = cur_radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos = np.array([x, y, z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)  # [num_views, 4, 4]

    # Expand from [num_views, *] to [batch_size, num_views, *]
    fxfycxcy = fxfycxcy[None, ...].repeat(batch_size, axis=0) # [batch_size, num_views, 4]
    c2ws = c2ws[None, ...].repeat(batch_size, axis=0)
    return {
        "w": w,
        "h": h,
        "num_views": num_views,
        "fxfycxcy": torch.from_numpy(fxfycxcy).to(device).float(),
        "c2w": torch.from_numpy(c2ws).to(device).float(),
    }


def get_interpolated_cameras(
    cameras,
    num_views,
):
    """
    For each consecutive pair of cameras, add num_views linearly interpolated views.
    """
    fxfycxcy = cameras['fxfycxcy']  # [batch_size, num_input_views, 4]
    c2w = cameras['c2w']  # [batch_size, num_input_views, 4, 4]
    
    batch_size, num_input_views = fxfycxcy.shape[:2]
    
    interpolated_fxfycxcy = []
    interpolated_c2w = []
    
    for b in range(batch_size):
        batch_fxfycxcy = []
        batch_c2w = []
        
        for i in range(num_input_views - 1):
            # Add the current view
            batch_fxfycxcy.append(fxfycxcy[b, i])
            batch_c2w.append(c2w[b, i])
            
            curr_fxfycxcy = fxfycxcy[b, i]
            next_fxfycxcy = fxfycxcy[b, i + 1]
            curr_c2w = c2w[b, i]
            next_c2w = c2w[b, i + 1]
            
            # Create alpha values for all interpolations at once
            alphas = torch.linspace(1 / (num_views + 1), num_views / (num_views + 1), num_views, device=fxfycxcy.device)
            
            # Batch interpolation for camera intrinsics
            interp_fxfycxcy = (1 - alphas[:, None]) * curr_fxfycxcy[None, :] + alphas[:, None] * next_fxfycxcy[None, :]
            batch_fxfycxcy.extend(interp_fxfycxcy)
            
            # Batch interpolation for camera poses
            # For rotation, we should use SLERP, but for simplicity using linear interpolation
            interp_c2w = (1 - alphas[:, None, None]) * curr_c2w[None, :, :] + alphas[:, None, None] * next_c2w[None, :, :]
            batch_c2w.extend(interp_c2w)
        
        # Add the last view
        batch_fxfycxcy.append(fxfycxcy[b, -1])
        batch_c2w.append(c2w[b, -1])
        
        interpolated_fxfycxcy.append(torch.stack(batch_fxfycxcy))
        interpolated_c2w.append(torch.stack(batch_c2w))
    
    return {
        'fxfycxcy': torch.stack(interpolated_fxfycxcy),
        'c2w': torch.stack(interpolated_c2w)
    }



parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml")
parser.add_argument("--load", type=str, default="weight/obj_res256.pt")
parser.add_argument("--data_path", type=str, default="data_example/dl3dv_sample_data_path.json")
parser.add_argument("--best_path", type=str, default="data_example/dl3dv_sample_best_path.json")
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--num_input_views", type=int, default=48)
parser.add_argument("--scene_inference", action="store_true")
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")

parser.add_argument("--max_yaw_deg", type=int, default=10)
parser.add_argument("--max_pitch_deg", type=int, default=10)
parser.add_argument("--num_yaw", type=int, default=5)
parser.add_argument("--num_pitch", type=int, default=5)
parser.add_argument("--max_rectify_deg", type=float, default=None)

args = parser.parse_args()
model_config = omegaconf.OmegaConf.load(args.config)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Create output directory if specified
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Seed everything
seed = 95
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model = LaCTLVSM(**model_config).cuda()

# Load checkpoint
print(f"Loading checkpoint from {args.load}...")
checkpoint = torch.load(args.load, map_location="cpu")
model.load_state_dict(checkpoint["model"])

# Data
dataset = NVSDataset(
    args.data_path, 
    args.num_input_views, 
    tuple(args.image_size), 
    sorted_indices=args.scene_inference, 
    scene_pose_normalize=args.scene_inference,
    best_path=args.best_path,
    max_rectify_deg=args.max_rectify_deg,
)
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(seed)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
)


for sample_idx, data_dict in enumerate(dataloader):
    scene_id = data_dict["scene_id"][0]  # batch size == 1
    name_best = [_[0] for _ in data_dict["name_best"]]  # batch size == 1
    image_best_raw = data_dict.pop("image_best_raw").cuda()
    fxfycxcy_best = data_dict.pop("fxfycxcy_best")  # [1, N, 4]
    c2w_best_raw = data_dict.pop("c2w_best_raw")  # [1, N, 4, 4]
    c2w_best_rectify = data_dict.pop("c2w_best_rectify")  # [1, N, 4, 4]
    deg_best_rectify = data_dict.pop("deg_best_rectify")  # [1, N]
    assert c2w_best_raw.shape == c2w_best_rectify.shape
    num_best_raw = c2w_best_raw.shape[1]
    num_best_rectify = c2w_best_rectify.shape[1]
    assert num_best_raw == len(name_best) == deg_best_rectify.shape[1]
    
    for idx_best in range(num_best_raw):
        name_best_one = os.path.splitext(os.path.basename(name_best[idx_best]))[0]
        save_dir = os.path.join(output_dir, scene_id, name_best_one)
        if os.path.exists(save_dir):
            image_list = [_ for _ in os.listdir(save_dir) if _.endswith(".jpg")]
            # +3: raw, raw_target, rectify; -1: maybe not save yaw=0 & pitch=0
            num_save_min = args.num_yaw * args.num_pitch + 3 - 1
            if len(image_list) >= num_save_min:
                print(f"Skipping {save_dir} because it already has {len(image_list)} images")
                continue                
        os.makedirs(save_dir, exist_ok=True)

        image_best_raw_one = image_best_raw[:, idx_best].unsqueeze(1)  # [1, 1, 3, H, W]
        fxfycxcy_best_one = fxfycxcy_best[:, idx_best].squeeze(0)  # [4,]
        c2w_best_raw_one = c2w_best_raw[:, idx_best].squeeze(0)  # [4, 4]
        c2w_best_rectify_one = c2w_best_rectify[:, idx_best].squeeze(0)  # [4, 4]
        deg_best_rectify_one = deg_best_rectify[:, idx_best].squeeze(0).item()  # [1,]

        c2w_nearbest_rectify = []
        fxfycxcy_nearbest = []
        yaw_list = []
        pitch_list = []
        for yaw_delta in np.linspace(-args.max_yaw_deg, args.max_yaw_deg, args.num_yaw):
            for pitch_delta in np.linspace(-args.max_pitch_deg, args.max_pitch_deg, args.num_pitch):
                if yaw_delta == 0 and pitch_delta == 0:
                    continue
                c2w_nearbest_rectify.append(
                    rotate_view_c2w(c2w_best_rectify_one, yaw_deg=yaw_delta, pitch_deg=pitch_delta, roll_deg=0)
                )  # list of [4, 4]
                fxfycxcy_nearbest.append(fxfycxcy_best_one)  # list of [4, ]
                yaw_list.append(yaw_delta)
                pitch_list.append(pitch_delta)

        num_nearbest = len(yaw_list)
        c2w_nearbest_rectify = torch.stack(c2w_nearbest_rectify).unsqueeze(0)  # [1, N, 4, 4]
        fxfycxcy_nearbest = torch.stack(fxfycxcy_nearbest).unsqueeze(0)  # [1, N, 4]
        fxfycxcy_best_one = fxfycxcy_best_one[None, None, :]  # [1, 1, 4]
        c2w_best_raw_one = c2w_best_raw_one[None, None, :]  # [1, 1, 4, 4]
        c2w_best_rectify_one = c2w_best_rectify_one[None, None, :]  # [1, 1, 4, 4]
        data_dict_best = {  
            "fxfycxcy": torch.cat(
                [fxfycxcy_best_one.cuda(), fxfycxcy_best_one.cuda(), fxfycxcy_nearbest.cuda()], 
                dim=1),
            "c2w": torch.cat(
                [c2w_best_raw_one.cuda(), c2w_best_rectify_one.cuda(), c2w_nearbest_rectify.cuda()], 
                dim=1),
        }

        data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
            rendering = model(data_dict, data_dict_best)
            target = image_best_raw_one
            rendering_raw = rendering[:, :1]
            psnr = -10.0 * torch.log10(F.mse_loss(rendering_raw, target)).item()
            print(f"Sample {sample_idx}: PSNR = {psnr:.2f}")
            
            # Save rendered images if output directory is specified
            if output_dir:
                def save_image_rgb(tensor, filepath):
                    """Save tensor as jpg image."""
                    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
                    numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
                    Image.fromarray(numpy_image, mode='RGB').save(filepath, quality=95, optimize=True)

                batch_size, num_views = rendering.shape[:2]
                assert batch_size == 1

                for batch_idx in range(batch_size):
                    # Save raw
                    save_image_rgb(rendering[batch_idx, 0], os.path.join(save_dir, "raw_render.jpg"))
                    save_image_rgb(target[batch_idx, 0], os.path.join(save_dir, "raw_target.jpg"))
                    # Save rectify
                    save_image_rgb(rendering[batch_idx, 1], os.path.join(save_dir, f"rectify_deg{deg_best_rectify_one:.2f}.jpg"))
                    # Save shift
                    num_render = rendering.shape[1]
                    assert num_render == num_nearbest + 2  # +2: raw, rectify
                    for idx_nearbest in range(num_nearbest):
                        view_idx = idx_nearbest + 2
                        yaw = round(yaw_list[idx_nearbest], 2)
                        pitch = round(pitch_list[idx_nearbest], 2)
                        filename = f"rectify_yaw{yaw:.2f}_pitch{pitch:.2f}.jpg"
                        save_image_rgb(rendering[batch_idx, view_idx], os.path.join(save_dir, filename))

                print(f"Saved images for sample {sample_idx} to {save_dir}")

        torch.cuda.empty_cache()
