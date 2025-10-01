import torch
from torch.utils.data import Dataset
import os
import os.path as osp
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2
import hashlib
import json
import time
import logging
import pdb
# from temp_utils import read_img, read_disp
from utils.file_io import read_img, read_disp
# import video_transforms
from . import video_transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoStereoDataset(Dataset):
    def __init__(self,
                 transform=None,
                 sequence_length=3, # Number of frames per sample
                 subsample_groundtruth_spring=True, # Spring provides GT at 4x superresolution
                 ):
        super(VideoStereoDataset, self).__init__()
        self.transform = transform
        self.sequence_length = sequence_length
        self.subsample_groundtruth_spring = subsample_groundtruth_spring
        
        # self.samples will be a list of lists. 
        # Each inner list contains dicts of file paths for one full video sequence.
        # e.g., [ [seq1_frame0_paths, seq1_frame1_paths, ...], [seq2_frame0_paths, ...] ]
        self.samples = []
        
        # self.valid_starts will be a list of tuples (sequence_index, frame_index)
        # indicating all valid starting points for a clip.
        self.valid_starts = []

    def __getitem__(self, index):
        # Get the start of the clip
        seq_idx, frame_idx = self.valid_starts[index]
        
        # Get the file paths for the entire clip
        clip_paths = self.samples[seq_idx][frame_idx : frame_idx + self.sequence_length]

        # --- Load all data for the sequence ---
        left_images, right_images, disparities, poses, intrinsics = [], [], [], [], []

        for frame_paths in clip_paths:
            # Load images and disparity
            left_images.append(read_img(frame_paths['left']))
            right_images.append(read_img(frame_paths['right']))
            
            if 'disp' in frame_paths:
                disp = read_disp(frame_paths['disp'])
                if self.subsample_groundtruth_spring:
                    disp = disp[::2,::2]
                disparities.append(disp)

            intrinsics.append(frame_paths['intrinsics'])
            poses.append(frame_paths['pose'])

        # --- Collect everything into a single sample dictionary ---
        sample = {
            'left': left_images,        # Shape: [[C, H, W]...]
            'right': right_images,      # Shape: [[C, H, W]...]
            'disp': disparities,        # Shape: [[H, W]...]
            'pose': poses,              # Shape: [[4, 4]...]
            'intrinsics': intrinsics # Usually the same for the whole sequence
        }

        # --- Apply transformations ---
        if self.transform:
            sample = self.transform(sample)

        # # --- Stack the frames into a single tensor ---
        # sample = {
        #     'left': left_images,        # Shape: [N, C, H, W]
        #     'right': right_images,      # Shape: [N, C, H, W]
        #     'disp': disparities,        # Shape: [N, H, W]
        #     'pose': poses,              # Shape: [N, 4, 4]
        #     'intrinsics': intrinsics[0] # Usually the same for the whole sequence
        # }

        return sample

    def __len__(self):
        return len(self.valid_starts)
        
    def build_valid_starts(self):
        """Populates the list of valid starting indices."""
        self.valid_starts = []
        for seq_idx, sequence in enumerate(self.samples):
            num_frames = len(sequence)
            if num_frames >= self.sequence_length:
                for frame_idx in range(num_frames - self.sequence_length + 1):
                    self.valid_starts.append((seq_idx, frame_idx))


class SpringDataset(VideoStereoDataset):
    def __init__(self, 
                 data_dir='/projects/NEUFR/data/Spring',
                 mode="train",
                 transform=None,
                 sequence_length=3,
                 ):
        # Initialize the parent VideoStereoDataset
        super(SpringDataset, self).__init__(transform=transform, sequence_length=sequence_length)
        
        baseline = 0.065 # Fixed for the whole spring dataset

        # --- 1. Find and Group all files by sequence ---
        seq_root = os.path.join(data_dir, mode)
        sequences = defaultdict(list)
        for seq in os.listdir(seq_root):
            all_left_files = sorted(glob(os.path.join(seq_root, seq, 'frame_left', '*.png')))

            intrinsics_list = []
            with open(os.path.join(seq_root, seq, "cam_data", "intrinsics.txt")) as f:
                for row in f:
                    intrinsics = torch.tensor([float(x) for x in row.split(" ")])
                    intrinsics_list.append(intrinsics)
            
            poses_list = []
            with open(os.path.join(seq_root, seq, "cam_data", "extrinsics.txt")) as f:
                for row in f:
                    pose1x16 = torch.tensor([float(x) for x in row.split(" ")])
                    pose4x4 = pose1x16.reshape(4,4)
                    poses_list.append(pose4x4)

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = left_path.replace("frame_left", "frame_right")
                sample['disp'] = left_path.replace("frame_left", "disp1_left").replace('.png', '.dsp5')
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                # sample['baseline'] = baseline

                sequences[seq].append(sample)

        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")



def build_dataset(args):

    if args.stage == 'spring':
        train_transform_list = [video_transforms.RandomScale(crop_width=768),
                                video_transforms.RandomCrop(384, 768),
                                video_transforms.RandomColor(),
                                video_transforms.RandomVerticalFlip(),
                                video_transforms.ToTensor(),
                                video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = video_transforms.Compose(train_transform_list)
        spring = SpringDataset(transform=train_transform)
        train_dataset = spring

        return train_dataset
    
if __name__=="__main__":
    train_transform_list = [video_transforms.RandomScale(crop_width=768),
                                video_transforms.RandomCrop(384, 768),
                                video_transforms.RandomColor(),
                                video_transforms.RandomVerticalFlip(),
                                video_transforms.ToTensor(),
                                video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

    train_transform = video_transforms.Compose(train_transform_list)
    spring = SpringDataset(transform=train_transform)
    
    data = spring[0]
    print(data.keys())

    pdb.set_trace()