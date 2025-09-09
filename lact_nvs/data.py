import json
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import cal_ground_normal, rectify_c2w


def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (height, width) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
    return cropped_image, [new_fx, new_fy, new_cx, new_cy]


def normalize(x):
    return x / x.norm()

def normalize_with_mean_pose(c2ws: torch.Tensor):
    # This is a historical code for scene camera normalization;
    #  thanks to the authors (might mostly credit to Zexiang Xu)

    # Get the mean parameters
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].mean(0)
    up = c2ws[:, :3, 1].mean(0)

    # Get the view matrix.
    vec2 = normalize(vec2)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, center], 1)

    # Extend the view matrix to 4x4.
    avg_pos = c2ws.new_zeros(4, 4)
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = m

    # Align coordinate system to the mean camera
    c2ws = torch.linalg.inv(avg_pos) @ c2ws

    # Scale the scene to the range of [-1, 1].
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale

    return c2ws


class NVSDataset(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
        best_path=None,
        max_rectify_deg=None,
    ):
        """
        image_size is (h, w) or just a int (as size).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.best_data_point_paths = None
        if best_path is not None:
            self.best_data_point_paths = json.load(open(best_path, "r"))
        self.max_rectify_deg = max_rectify_deg
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize

        self.num_views = num_views
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        scene_id = os.path.basename(os.path.dirname(data_point_path))
        with open(data_point_path, "r") as f:
            images_info = json.load(f)
        
        print("=" * 100)
        print("Processing scene:", scene_id)
        ############################################################
        # 1. 对所有图像的c2ws进行normalize_with_mean_pose
        c2w_all = []
        for info in images_info:
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_all.append(c2w)
        c2ws = torch.stack(c2w_all)
        if self.scene_pose_normalize:
            print("Normalizing scene poses...")
            c2ws = normalize_with_mean_pose(c2ws)
        for idx, c2w in enumerate(c2ws):
            images_info[idx]["c2w"] = c2w

        ############################################################
        # 2. 找到与地面垂直的normal
        vectors = []
        for idx, c2w in enumerate(c2ws):
            vectors.append(c2w[:3, 3].numpy())
        normal, r2, _, line_like, lambdas = cal_ground_normal(np.array(vectors))
        normal = torch.tensor(normal)
        print("The R^2 of the ground normal fitting:", r2, "lambdas:", lambdas, "line_like:", line_like)

        ############################################################
        # 3. 采样出训练图像 + best图像
        # If the num_views is larger than the number of images, use all images
        indices = random.sample(range(len(images_info)), min(self.num_views, len(images_info)))
        if self.sorted_indices:
            indices = sorted(indices)

        best_index_list = []
        if self.best_data_point_paths is not None:
            best_data_point_path = os.path.join(self.base_dir, self.best_data_point_paths[index])
            with open(best_data_point_path, "r") as f:
                best_list = json.load(f)
            for index, info in enumerate(images_info):
                if info["file_path"] in best_list:
                    best_index_list.append(index)
        num_best = len(best_index_list)
        indices = indices + best_index_list

        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        
        for index in indices:
            info = images_info[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]

            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            image = Image.open(image_path)
            
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)

            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')
            
            c2w_list.append(info["c2w"])
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))

        ############################################################
        # 4. 对best图像进行rectify
        c2w_best_rectify_list = []
        deg_best_rectify_list = []
        for index in best_index_list:
            info = images_info[index]
            if line_like:
                c2w_rectify = info["c2w"]
                rectify_deg = 0
            else:
                c2w_rectify, rectify_deg = rectify_c2w(info["c2w"], normal, self.max_rectify_deg)
            c2w_best_rectify_list.append(c2w_rectify)
            deg_best_rectify_list.append(rectify_deg)

        return {
            "scene_id": scene_id,
            "fxfycxcy": torch.tensor(fxfycxcy_list[:-num_best]),
            "c2w": torch.stack(c2w_list[:-num_best]),
            "image": torch.stack(image_list[:-num_best]),
            "fxfycxcy_best": torch.tensor(fxfycxcy_list)[-num_best:],
            "c2w_best_raw": torch.stack(c2w_list)[-num_best:],
            "image_best_raw": torch.stack(image_list)[-num_best:],
            "c2w_best_rectify": torch.stack(c2w_best_rectify_list),
            "deg_best_rectify": torch.tensor(deg_best_rectify_list),
            "name_best": best_list,
        }
