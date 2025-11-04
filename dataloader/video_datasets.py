import torch
from torch.utils.data import Dataset
import os
import os.path as osp
from collections import defaultdict
import numpy as np
import random
from torch.utils.data import Dataset
from glob import glob
import cv2
import hashlib
import json
import time
import logging
import pdb
from typing import Iterable
# from temp_utils import read_img, read_disp
from utils.file_io import read_img, read_disp
# import video_transforms
from . import video_transforms
from . import datasets as single_datasets


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
        # Default behaviour: use the fixed sequence_length and the precomputed
        # valid_starts (this keeps backwards-compatibility for datasets that
        # expect fixed-length clips).
        seq_idx, frame_idx = self.valid_starts[index]
        return self.get_clip(seq_idx, frame_idx, self.sequence_length)

    def get_clip(self, seq_idx, frame_idx, length):
        """
        Load a clip starting at (seq_idx, frame_idx) with the requested length.
        This helper is used by LengthAugmentedDataset to fetch clips of arbitrary
        lengths (up to the available maximum for that start).
        """
        clip_paths = self.samples[seq_idx][frame_idx: frame_idx + length]

        # --- Load all data for the sequence ---
        left_images, right_images, disparities, poses, intrinsics = [], [], [], [], []

        for frame_paths in clip_paths:
            # Load images and disparity
            left_images.append(read_img(frame_paths['left']))
            right_images.append(read_img(frame_paths['right']))

            if 'disp' in frame_paths and frame_paths['disp'] is not None:
                disp = read_disp(frame_paths['disp'])
                if self.subsample_groundtruth_spring:
                    disp = disp[::2, ::2]
                disparities.append(disp)

            intrinsics.append(frame_paths.get('intrinsics'))
            poses.append(frame_paths.get('pose'))

        sample = {
            'left': left_images,
            'right': right_images,
            'disp': disparities,
            'pose': poses,
            'intrinsics': intrinsics
        }

        if self.transform:
            sample = self.transform(sample)

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

    def build_all_starts(self):
        """
        Build all possible start indices and the maximum available length from each start.
        Populates:
            - self.valid_starts_all: list of (seq_idx, frame_idx)
            - self.max_len_per_start: list of ints (same length)
        This is used by LengthAugmentedDataset to support batching by different
        requested sequence lengths.
        """
        self.valid_starts_all = []
        self.max_len_per_start = []
        for seq_idx, sequence in enumerate(self.samples):
            num_frames = len(sequence)
            for frame_idx in range(num_frames):
                self.valid_starts_all.append((seq_idx, frame_idx))
                self.max_len_per_start.append(num_frames - frame_idx)


"""Video wrappers for existing single-frame stereo datasets.

The primary helper here is VideoFromStereo which accepts an instance of a
`StereoDataset` (or any object with a `samples` list of dicts) and exposes the
`VideoStereoDataset` API by grouping samples into sequences.

Usage examples:
    base = KITTI15(...)
    video_ds = VideoFromStereo(base, sequence_length=1)  # each sample becomes a 1-frame clip

    # or group by parent scene folder (useful when the underlying dataset stores
    # multiple frames per scene in a folder)
    video_ds = VideoFromStereo(base, sequence_length=3, grouping='by_parent', group_depth=2)

Design notes:
- If grouping='none' (default) each input sample becomes its own 1-length
  sequence. This is the safest option and preserves exact behaviour of
  single-frame datasets when sequence_length==1.
- grouping='by_parent' will group samples that share the same parent directory
  at `group_depth` levels above the file path. This is a heuristic and works
  well for datasets that store frames per-scene in a folder.

The wrapper does not attempt to change or run existing transforms attached to
the base dataset; instead you can pass a `transform` to VideoFromStereo which
will be used for video-aware transforms. If you want to reuse the base
dataset's transform for single-frame behaviour, pass `transform=base.transform`.
"""

class VideoFromStereo(VideoStereoDataset):
    def __init__(self,
                 base_dataset,
                 sequence_length: int = 1,
                 grouping: str = 'none',
                 group_depth: int = 2,
                 transform=None,
                 subsample_groundtruth_spring: bool = True,
                 ):
        """
        Wrap a single-frame stereo dataset into a video-style dataset.

        Args:
            base_dataset: an instance that exposes `samples` (list of dicts).
            sequence_length: desired number of frames per clip (default 1).
            grouping: 'none' or 'by_parent'. If 'by_parent' samples that share
                the same parent path at `group_depth` levels will be grouped.
            group_depth: how many levels above the file path to use as the
                grouping key when grouping='by_parent'. Default 2.
            transform: optional transform to apply to the video sample (overrides base).
            subsample_groundtruth_spring: forwarded to parent for compatibility.
        """

        super(VideoFromStereo, self).__init__(transform=transform,
                                              sequence_length=sequence_length,
                                              subsample_groundtruth_spring=subsample_groundtruth_spring)

        # Validate base_dataset
        if not hasattr(base_dataset, 'samples'):
            raise ValueError("base_dataset must have a `samples` attribute (list of dicts)")

        base_samples = base_dataset.samples

        # If base already stores sequences (list of lists), adopt directly
        if len(base_samples) > 0 and isinstance(base_samples[0], list):
            self.samples = base_samples
        else:
            # base_samples is a flat list of dicts -> convert to sequences
            if grouping == 'none':
                self.samples = [[s] for s in base_samples]
            elif grouping == 'by_parent':
                groups = defaultdict(list)
                for s in base_samples:
                    left = s.get('left') or s.get('left_name') or ''
                    # compute parent key by ascending `group_depth` levels
                    key = left
                    for _ in range(group_depth):
                        key = osp.dirname(key)
                    groups[key].append(s)

                # sort groups by key for deterministic ordering, and ensure
                # frames within each group are sorted by filename
                self.samples = []
                for key in sorted(groups.keys()):
                    frames = sorted(groups[key], key=lambda x: x.get('left') or '')
                    self.samples.append(frames)
            else:
                raise ValueError(f"Unknown grouping mode: {grouping}")

        # Build valid starts for sliding-window indexing
        self.build_valid_starts()

    def __repr__(self):
        return f"<VideoFromStereo samples={len(self.samples)} sequence_length={self.sequence_length} valid_clips={len(self.valid_starts)}>"


class SpringDataset(VideoStereoDataset):
    def __init__(self, 
                 data_dir='/projects/NEUFR/data/Spring',
                 mode="train",
                 transform=None,
                 sequence_length=3,
                 random_sequence=False,
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
        # For spring we might want variable-length clips. If random_sequence is
        # enabled, we add every frame as a valid start (the length will be
        # sampled randomly at __getitem__). Otherwise use fixed length behaviour.
        self.random_sequence = bool(random_sequence)
        if self.random_sequence:
            # include every possible start
            self.valid_starts = []
            for seq_idx, sequence in enumerate(self.samples):
                num_frames = len(sequence)
                for frame_idx in range(num_frames):
                    self.valid_starts.append((seq_idx, frame_idx))
        else:
            self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

    def __getitem__(self, index):
        """
        Override to support random-length clips when `random_sequence` is True.
        If random_sequence is enabled, sample a length uniformly between 1 and
        the maximum possible length from the chosen start.
        """
        seq_idx, frame_idx = self.valid_starts[index]

        # If random_sequence, compute the maximum available length from this start
        if getattr(self, 'random_sequence', False):
            max_len = len(self.samples[seq_idx]) - frame_idx
            # sample a random length between 1 and max_len (inclusive)
            chosen_len = random.randint(1, max_len)
        else:
            chosen_len = self.sequence_length

        clip_paths = self.samples[seq_idx][frame_idx: frame_idx + chosen_len]

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

        return sample



def build_dataset(cfg):
    """
    Build a video-compatible dataset from config.

    cfg.stage may be a string or a list of stage names. Each stage will be
    instantiated (using single-frame dataset classes when available) and
    wrapped with VideoFromStereo so the returned dataset always exposes the
    VideoStereoDataset API. All datasets are concatenated into one dataset.

    Note: Use cfg.sequence_length (int) to force a global clip length. If not
    provided, defaults to 1 (single-frame) which is safe when combining
    multiple datasets. If you only train on a video dataset like 'spring', set
    cfg.sequence_length to the desired clip length (e.g. 3).
    """

    # Normalize stages to a list
    stages = cfg.stage if isinstance(cfg.stage, (list, tuple)) else [cfg.stage]

    # Global sequence length (default 1). You can override in cfg.
    if hasattr(cfg, 'get'):
        sequence_length = int(cfg.get('sequence_length', 1))
    else:
        sequence_length = getattr(cfg, 'sequence_length', 1)

    # Compose a default video transform (shared across datasets)
    train_transform_list = [video_transforms.RandomScale(crop_width=768),
                            video_transforms.RandomCrop(384, 768),
                            video_transforms.RandomColor(),
                            video_transforms.RandomVerticalFlip(),
                            video_transforms.ToTensor(),
                            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]

    train_transform = video_transforms.Compose(train_transform_list)

    combined_dataset = None
    combined_seq_source = []

    for stage in stages:
        if stage is None or stage == '':
            continue

        s = str(stage).lower()

        if s == 'spring':
            # For spring we want random-length clips between 1 and the max
            # available for each sequence.
            ds = SpringDataset(transform=train_transform, random_sequence=True)

        elif s in ('foundation', 'fsd'):
            base = single_datasets.FoundationStereo(transform=None)
            # stereo image datasets should use sequence length 1
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        elif s in ('kitti15','kitti_15'):
            base = single_datasets.KITTI15(transform=None)
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        elif s in ('kitti12','kitti_12'):
            base = single_datasets.KITTI12(transform=None)
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        elif s == 'driving':
            base = single_datasets.Driving(transform=None)
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        elif s == 'monkaa':
            base = single_datasets.Monkaa(transform=None)
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        elif s == ('flyingthings3d'):
            base = single_datasets.FlyingThings3D(transform=None)
            ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)

        else:
            # Fallback: try to create an attribute from single_datasets with the same name
            try:
                cls = getattr(single_datasets, stage)
                base = cls(transform=None)
                ds = VideoFromStereo(base, sequence_length=1, grouping='none', transform=train_transform)
            except Exception as e:
                logging.warning(f"Unknown stage '{stage}' - skipping. Error: {e}")
                continue

        if combined_dataset is None:
            combined_dataset = ds
        else:
            # extend combined samples
            combined_dataset.samples.extend(ds.samples)
        # record source (stage) for each sequence added from this dataset
        combined_seq_source.extend([s] * len(ds.samples))

    if combined_dataset is None:
        raise ValueError(f"No valid datasets found for stages: {stages}")

    combined_dataset.sequence_length = sequence_length
    combined_dataset.transform = train_transform
    # Attach per-sequence source labels so samplers can create homogeneous batches
    combined_dataset.seq_source = combined_seq_source
    combined_dataset.build_valid_starts()

    return combined_dataset
    
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