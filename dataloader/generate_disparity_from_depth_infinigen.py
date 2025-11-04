import torch
import os
import os.path as osp
import numpy as np
from glob import glob
import pdb
from PIL import Image
from tqdm import tqdm

class InfinigenSV():
    def __init__(self,
                 data_dir='/projects/NEUFR/data/Infinigen',
                 mode="train",
                 ):

        self.depth_eps = 1e-5

        # --- 1. Find and Group all files by sequence ---
        seq_root = os.path.join(data_dir, mode)
        for seq in os.listdir(seq_root):
            all_left_files = sorted(glob(os.path.join(seq_root, seq, 'frames/Image/camera_0', '*.png')))
            
            # Get the intrinsics and poses
            all_camera_data_path_left = sorted(glob(os.path.join(seq_root, seq, 'frames/camview/camera_0', "*.npz")))
            print(len(all_camera_data_path_left))
            intrinsics_list = []
            poses_list = []
            for intrinsics_path in all_camera_data_path_left:
                data_left = np.load(intrinsics_path)
                data_right = np.load(intrinsics_path.replace('camera_0', 'camera_1').replace("0.npz", "1.npz"))
                K = data_left['K']
                T_left = data_left['T']
                T_right = data_right['T']
                intrinsics = torch.tensor([K[0,0], K[1,1], K[0,2], K[1,2]])
                intrinsics_list.append(intrinsics)
                pose = torch.tensor(T_left)
                poses_list.append(pose)

            # Get focal length in pixels
            focal_length_px = K[0,0]
            
            # Compute baseline only once
            baseline = np.linalg.norm(T_left[:3, 3] - T_right[:3, 3])

            # Get depth2disp scale
            depth2disp_scale = focal_length_px * baseline

            # Read depth maps and convert to disparity
            for left_path in tqdm(all_left_files):
                depth_path = left_path.replace("Image", "Depth").replace(".png", ".npy")
                depth_image = np.load(depth_path)
                depth_mask = depth_image < self.depth_eps
                depth_image[depth_mask] = self.depth_eps
                disp = depth2disp_scale / depth_image
                disp[depth_mask] = 0
                valid_disp = (disp < 512) * (1 - depth_mask)
                disp = np.array(disp).astype(np.float32)

                # Save disparity
                save_path = left_path.replace("Image", "Disparity").replace(".png", ".npy")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, disp)
    
if __name__=="__main__":
    dataset = InfinigenSV()
