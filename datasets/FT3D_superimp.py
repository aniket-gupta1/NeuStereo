import cv2
import torch.nn.functional as F
import numpy as np
from default import FlowDataset
import glob
import os.path.join as osp
import frame_utils
import torch
import random

class FlyingThings3DSuperimposed(FlowDataset):
    def __init__(self, aug_params=None,
                 root='/projects/nufr/aniket/Datasets/Stereo_Disp/FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=False,
                 only_left=False,
                 superimpose_mode='blend',  # New parameter
                 rectify=False,  # Whether to rectify before superimposing
                 ):
        super(FlyingThings3DSuperimposed, self).__init__(aug_params)

        self.superimpose_mode = superimpose_mode
        self.rectify = rectify
        
        img_dir = root
        flow_dir = root

        if only_left:
            cam_list = ['left']
        else: 
            cam_list = ['left', 'right']

        if test_set or validate_subset:
            split = 'TEST'
        else:
            split = 'TRAIN'
        
        left_images = sorted(glob(osp.join(root, dstype, split, '*/*/left/*.npy')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(dstype, 'disparity').replace('.npy', '.pfm') for im in left_images]

        # If rectifying, we'll need camera parameters
        if self.rectify:
            # Load or define camera parameters for FT3D
            # These are typically the same for all frames in FT3D
            self.setup_rectification_params()

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [[img1, img2]]
                self.disp_list += [disp]
    
    def setup_rectification_params(self):
        """Setup rectification parameters for FT3D"""
        # FT3D camera parameters (these are approximate - adjust based on actual params)
        focal_length = 1050.0
        cx, cy = 479.5, 269.5
        baseline = 10.0  # cm
        
        # Intrinsic matrices (same for both cameras in FT3D)
        self.K = np.array([[focal_length, 0, cx],
                          [0, focal_length, cy],
                          [0, 0, 1]])
        
        # No distortion in synthetic data
        self.D = np.zeros(5)
        
        # Rotation and translation between cameras
        # For FT3D, cameras are roughly aligned but not perfectly
        self.R = np.eye(3)  # Small rotation
        self.T = np.array([baseline, 0, 0]).reshape(3, 1)
        
    def rectify_pair(self, left_img, right_img):
        """Rectify stereo pair"""
        h, w = left_img.shape[:2]
        
        # Get rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K, self.D,
            self.K, self.D,
            (w, h),
            self.R, self.T,
            alpha=0
        )
        
        # Create rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.K, self.D, R1, P1, (w, h), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.K, self.D, R2, P2, (w, h), cv2.CV_32FC1
        )
        
        # Rectify
        left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect
    
    def superimpose_images(self, img1, img2, mode='blend'):
        """
        Superimpose stereo images using various methods
        
        Args:
            img1, img2: numpy arrays of shape (H, W, 3)
            mode: superimposition method
        
        Returns:
            superimposed: tensor ready for network input
        """
        if mode == 'blend':
            # Simple averaging
            superimposed = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
            
        elif mode == 'blend_diff':
            # Blend + difference channels
            blend = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
            diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
            signed_diff = (img1.astype(np.float32) - img2.astype(np.float32)) / 2.0 + 127.5
            
            # Stack as 9-channel input
            superimposed = np.concatenate([blend, diff, signed_diff], axis=-1)
            
        elif mode == 'stack':
            # Stack left and right as 6 channels
            superimposed = np.concatenate([img1, img2], axis=-1)
            
        elif mode == 'stack_all':
            # Stack everything for maximum information
            blend = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
            diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
            
            superimposed = np.concatenate([
                img1.astype(np.float32),           # Original left (3)
                img2.astype(np.float32),           # Original right (3)
                blend,                              # Blended (3)
                diff,                               # Absolute diff (3)
            ], axis=-1)  # Total: 12 channels
            
        elif mode == 'anaglyph':
            # Red-cyan anaglyph style
            superimposed = np.zeros_like(img1, dtype=np.float32)
            superimposed[:, :, 0] = img2[:, :, 0]  # Red from right
            superimposed[:, :, 1] = img2[:, :, 1]  # Green from right
            superimposed[:, :, 2] = img1[:, :, 2]  # Blue from left
            
        elif mode == 'interlace':
            # Column interlacing
            superimposed = img1.astype(np.float32).copy()
            superimposed[:, 1::2] = img2[:, 1::2]
            
        elif mode == 'checkerboard':
            # Checkerboard pattern
            block_size = 8
            superimposed = img1.astype(np.float32).copy()
            for i in range(0, img1.shape[0], block_size):
                for j in range(0, img1.shape[1], block_size):
                    if (i//block_size + j//block_size) % 2 == 1:
                        superimposed[i:i+block_size, j:j+block_size] = \
                            img2[i:i+block_size, j:j+block_size]
        
        elif mode == 'phase':
            # Phase-based encoding (experimental)
            # Convert to grayscale for phase
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Compute phase correlation
            fft1 = np.fft.fft2(gray1)
            fft2 = np.fft.fft2(gray2)
            cross_power = (fft1 * np.conj(fft2)) / (np.abs(fft1 * np.conj(fft2)) + 1e-8)
            phase_corr = np.fft.ifft2(cross_power).real
            
            # Combine with original images
            phase_corr = np.stack([phase_corr] * 3, axis=-1)
            superimposed = np.concatenate([
                img1.astype(np.float32),
                img2.astype(np.float32),
                phase_corr * 255
            ], axis=-1)
            
        else:
            raise ValueError(f"Unknown superimpose mode: {mode}")
            
        return superimposed

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            
            # Optionally rectify
            if self.rectify:
                img1, img2 = self.rectify_pair(img1, img2)
            
            # Superimpose
            superimposed = self.superimpose_images(img1, img2, self.superimpose_mode)
            
            # Convert to tensor
            superimposed = torch.from_numpy(superimposed).permute(2, 0, 1).float()
            
            # Also return original images for visualization/validation
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            
            return superimposed, img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = frame_utils.read_gen(self.disp_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        # Handle grayscale
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # Apply augmentation BEFORE superimposition
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        
        # Optionally rectify (note: this changes disparity values!)
        if self.rectify:
            img1, img2 = self.rectify_pair(img1, img2)
            # WARNING: Disparity ground truth may need adjustment here!
            # For now, we're using unrectified disparity - this is a simplification
        
        # Superimpose the augmented images
        superimposed = self.superimpose_images(img1, img2, self.superimpose_mode)
        
        # Convert to tensors
        superimposed = torch.from_numpy(superimposed).permute(2, 0, 1).float()
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        flow = flow[:1]  # Only horizontal disparity
        
        # Return superimposed as main input, keep original images for visualization
        return {
            'superimposed': superimposed,
            'left': img1_tensor,
            'right': img2_tensor,
            'disparity': flow,
            'valid': valid.float()
        }