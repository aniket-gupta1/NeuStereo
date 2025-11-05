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
                 sequence_length=1, # Number of frames per sample
                 subsample_groundtruth_2x=False, # Spring provides GT at 4x (2x * 2x) superresolution
                 is_middlebury_eth3d=False,
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
        left_images, right_images, disparities, poses, intrinsics, baselines = [], [], [], [], [], []

        for frame_paths in clip_paths:
            # Load images and disparity
            left_images.append(read_img(frame_paths['left']))
            right_images.append(read_img(frame_paths['right']))
            
            disp = None
            if 'disp' in frame_paths and frame_paths['disp'] is not None:
                disp = read_disp(frame_paths['disp'])
                if disp is not None:
                    assert (disp >= 0).all(), f"Negative disparity found in {frame_paths['disp']}"
                    if self.subsample_groundtruth_spring:
                        disp = disp[::2,::2]
            disparities.append(disp)

            intrinsics.append(frame_paths['intrinsics'])
            poses.append(frame_paths['pose'])
            baselines.append(frame_paths['baseline'])

        # --- Collect everything into a single sample dictionary ---
        sample = {
            'left': left_images,        # Shape: [[C, H, W]...]
            'right': right_images,      # Shape: [[C, H, W]...]
            'disp': disparities,        # Shape: [[H, W]...]
            'pose': poses,              # Shape: [[4, 4]...]
            'intrinsics': intrinsics,    # Usually the same for the whole sequence
            'baseline': baselines
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

#region: Stereo Vidio Datasets ===================================================================
class SpringDataset(VideoStereoDataset):
    def __init__(self, 
                 data_dir='/projects/NEUFR/data/Spring',
                 mode="train",
                 transform=None,
                 sequence_length=2,
                 ):
        # Initialize the parent VideoStereoDataset
        super(SpringDataset, self).__init__(transform=transform, sequence_length=sequence_length, subsample_groundtruth_2x=True)
        
        baseline = torch.tensor(0.065) # Fixed for the whole spring dataset

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
                sample['baseline'] = baseline

                sequences[seq].append(sample)

        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

class InfinigenSV(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/NEUFR/data/Infinigen',
                 mode="train",
                 transform=None,
                 sequence_length=2,
                 ):
        # Initialize the parent VideoStereoDataset
        super(InfinigenSV, self).__init__(transform=transform, sequence_length=sequence_length)

        # --- 1. Find and Group all files by sequence ---
        seq_root = os.path.join(data_dir, mode)
        sequences = defaultdict(list)
        for seq in os.listdir(seq_root):
            all_left_files = sorted(glob(os.path.join(seq_root, seq, 'frames/Image/camera_0', '*.png')))
            all_camera_data_path_left = sorted(glob(os.path.join(seq_root, seq, 'frames/camview/camera_0', '*.npz')))
            
            if len(all_camera_data_path_left) == 0:
                print(f"Warning: No camera data found in {os.path.join(seq_root, seq, 'frames/camview/camera_0')}")
                continue
            
            # Compute baseline once from first frame
            try:
                first_data_left = np.load(all_camera_data_path_left[0])
                # Get corresponding right camera file by replacing last char before .npz
                first_right_path = all_camera_data_path_left[0].replace('_0.npz', '_1.npz')
                
                # If that doesn't exist, try finding it directly
                if not os.path.exists(first_right_path):
                    right_dir = os.path.dirname(all_camera_data_path_left[0]).replace('camera_0', 'camera_1')
                    right_files = sorted(glob(os.path.join(right_dir, '*.npz')))
                    if len(right_files) == 0:
                        raise FileNotFoundError(f"No camera 1 data found in {right_dir}")
                    first_right_path = right_files[0]
                
                first_data_right = np.load(first_right_path)
                T_left = first_data_left['T']
                T_right = first_data_right['T']
                baseline = torch.tensor(np.linalg.norm(T_left[:3, 3] - T_right[:3, 3]))
            except Exception as e:
                print(f"Error loading camera data for sequence {seq}: {str(e)}")
                continue
                
            intrinsics_list = []
            poses_list = []
            for intrinsics_path in all_camera_data_path_left:
                data_left = np.load(intrinsics_path)
                K = data_left['K']
                T_left = data_left['T']
                
                # Ensure pose is correctly formatted as a 4x4 transformation matrix
                pose = torch.eye(4)
                pose[:3, :3] = torch.from_numpy(T_left[:3, :3])  # Rotation
                pose[:3, 3] = torch.from_numpy(T_left[:3, 3])    # Translation
                
                intrinsics = torch.tensor([K[0,0], K[1,1], K[0,2], K[1,2]])
                intrinsics_list.append(intrinsics)
                poses_list.append(pose)
            
            # Compute baseline only once
            baseline = torch.tensor(np.linalg.norm(T_left[:3, 3] - T_right[:3, 3]))

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                # Fix the right image path by replacing camera directory, not file suffix
                right_path = left_path.replace('/camera_0/', '/camera_1/')
                right_path = right_path.replace('_0.png', '_1.png')
                sample['left'] = left_path
                sample['right'] = right_path
                sample['disp'] = left_path.replace("Image", "Disparity").replace(".png", ".npy")
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = baseline

                sequences[seq].append(sample)

        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

#endregion =======================================================================================

#region: Foundation Stereo Dataset ===============================================================
class FoundationStereo(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/NEUFR/data/FSD',
                 split_folders='all',
                 validate_subset=False,
                 max_scenes_per_split=None,
                 test_set=False,
                 use_cache=True,
                 transform=None,
                 sequence_length=1,
                 ):
        # Initialize the parent VideoStereoDataset
        super(FoundationStereo, self).__init__(transform=transform, sequence_length=sequence_length, is_FSD=True)

        self.data_dict = []

        # Generate cache filename
        cache_params = {
            'split_folders': split_folders,
            'max_scenes_per_split': max_scenes_per_split,
            'test_set': test_set,
            'validate_subset': validate_subset
        }

        cache_hash = hashlib.md5(str(cache_params).encode()).hexdigest()
        cache_file = f"{data_dir}/foundation_stereo_cache_{cache_hash}.json"

        # Try loading from cache FIRST
        if use_cache and os.path.exists(cache_file):
            success = self.load_from_cache(cache_file, cache_params)

        # Now the data_dict should be a list of dictionaries containing all file names, we can treat each one as a sequence
        sequences = defaultdict(list)
        for seq_id in range(len(self.data_dict)):
            all_left_files = [self.data_dict[seq_id]['left']]
            all_right_files = [self.data_dict[seq_id]['right']]
            all_disp_files = [self.data_dict[seq_id]['disp']]

            intrinsics_list = [torch.tensor([0.0, 0.0, 0.0, 0.0])]
            poses_list = [torch.eye(4)]
            baseline = torch.tensor(0.10)

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, right_path, disp_path, intrinsics, pose in zip(all_left_files, all_right_files, all_disp_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = right_path
                sample['disp'] = disp_path
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = baseline

                sequences[seq_id].append(sample)

        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")
        
    def load_from_cache(self, cache_file, cache_params):
        # logging.info(f"Loading dataset from cache: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache parameters match
            if cached_data.get('parameters') == cache_params:
                # Load the cached lists directly
                for (left, right), disparity in zip(cached_data['image_list'], cached_data['disp_list']):
                    sample = {
                        'left': left,
                        'right': right,
                        'disp': disparity
                    }
                    self.data_dict.append(sample)

                return True
            else:
                logging.info("Cache parameters don't match, rebuilding...")

        except Exception as e:
            logging.info(f"Cache loading failed: {e}. Rebuilding dataset...")
            return False

#endregion =======================================================================================

#region: SceneFlow Datasets ======================================================================
class Monkaa(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Monkaa',
                 split='frames_finalpass',
                 transform=None,
                 sequence_length=2,
                 ):
        # Initialize the parent VideoStereoDataset
        super(Monkaa, self).__init__(transform=transform, sequence_length=sequence_length)

        # For Monkaa, the intrinsics are same for the entire dataset
        intrinsics = torch.tensor([1050.0, 1050.0, 479.5, 269.5])

        # --- 1. Find and Group all files by sequence ---
        sequences = defaultdict(list)
        seq_root = os.path.join(data_dir, split)
        for seq in os.listdir(seq_root):
            all_left_files = sorted(glob(os.path.join(seq_root, seq, 'left', '*.png')))

            # For Monkaa, the intrinsics are same for the entire dataset
            intrinsics_list = [intrinsics for _ in range(len(all_left_files))]

            # Get poses from the camera data
            poses_list = []
            with open(os.path.join(data_dir, 'camera_data', seq, "camera_data.txt")) as f:
                for row in f:
                    if row.startswith('L'):
                        pose1x16 = torch.tensor([float(x) for x in row.split(" ")[1:]])
                        pose4x4 = pose1x16.reshape(4,4)
                        poses_list.append(pose4x4)

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 
        
            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = left_path.replace("left", "right")
                sample['disp'] = left_path.replace(split, "disparity").replace('.png', '.pfm')
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = torch.tensor(1.0)

                sequences[seq].append(sample)
        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

class Driving(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Driving',
                 split='frames_finalpass',
                 transform=None,
                 sequence_length=1,
                 ):
        # Initialize the parent VideoStereoDataset
        super(Driving, self).__init__(transform=transform, sequence_length=sequence_length)

        # For FlyingThings3D, the intrinsics are same for the entire dataset
        intrinsics_35 = torch.tensor([1050.0, 1050.0, 479.5, 269.5])
        intrinsics_15 = torch.tensor([450.0, 450.0, 479.5, 269.5])

        # --- 1. Find and Group all files by sequence ---
        subsets = ['15mm_focallength', '35mm_focallength']
        directions = ['scene_forwards', 'scene_backwards']

        sequences = defaultdict(list)
        for subset in subsets:
            if subset == '15mm_focallength':
                intrinsics = intrinsics_15
            else:
                intrinsics = intrinsics_35

            for direction in directions:
                seq_root = os.path.join(data_dir, split, subset, direction)
                for seq in os.listdir(seq_root):
                    all_left_files = sorted(glob(os.path.join(seq_root, seq, 'left', '*.png')))

                    # Get intrinsics
                    intrinsics_list = [intrinsics for _ in range(len(all_left_files))]

                    # Get poses from the camera data
                    poses_list = []
                    with open(os.path.join(data_dir, 'camera_data', subset, direction, seq, "camera_data.txt")) as f:
                        for row in f:
                            if row.startswith('L'):
                                pose1x16 = torch.tensor([float(x) for x in row.split(" ")[1:]])
                                pose4x4 = pose1x16.reshape(4,4)
                                poses_list.append(pose4x4)

                    assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 
                
                    for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                        sample = {}

                        sample['left'] = left_path
                        sample['right'] = left_path.replace("left", "right")
                        sample['disp'] = left_path.replace(split, "disparity").replace('.png', '.pfm')
                        sample['intrinsics'] = intrinsics
                        sample['pose'] = pose
                        sample['baseline'] = torch.tensor(1.0)

                        sequences[seq].append(sample)
        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

class FlyingThings3D(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/FlyingThings3D',
                 mode="train",
                 split='frames_finalpass',
                 transform=None,
                 sequence_length=2,
                 ):
        # Initialize the parent VideoStereoDataset
        super(FlyingThings3D, self).__init__(transform=transform, sequence_length=sequence_length)

        # For FlyingThings3D, the intrinsics are same for the entire dataset
        intrinsics = torch.tensor([1050.0, 1050.0, 479.5, 269.5])

        # --- 1. Find and Group all files by sequence ---
        subsets = ['A', 'B', 'C']
        sequences = defaultdict(list)
        for subset in subsets:
            seq_root = os.path.join(data_dir, split, mode, subset)
            for seq in os.listdir(seq_root):
                all_left_files = sorted(glob(os.path.join(seq_root, seq, 'left', '*.npy')))

                # For FlyingThings3D, the intrinsics are same for the entire dataset
                intrinsics_list = [intrinsics for _ in range(len(all_left_files))]

                # Get poses from the camera data
                poses_list = []
                with open(os.path.join(data_dir, 'camera_data', mode, subset, seq, "camera_data.txt")) as f:
                    for row in f:
                        if row.startswith('L'):
                            pose1x16 = torch.tensor([float(x) for x in row.split(" ")[1:]])
                            pose4x4 = pose1x16.reshape(4,4)
                            poses_list.append(pose4x4)

                assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 
            
                for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                    sample = {}

                    sample['left'] = left_path
                    sample['right'] = left_path.replace("left", "right")
                    sample['disp'] = left_path.replace(split, "disparity").replace('.png', '.pfm')
                    sample['intrinsics'] = intrinsics
                    sample['pose'] = pose
                    sample['baseline'] = torch.tensor(1.0)

                    sequences[seq].append(sample)
        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

#endregion =======================================================================================

#region: Evaluation Datasets =====================================================================
class InferenceDataset(VideoStereoDataset):
    def __init__(self,
                 data_dir = "/projects/nufr/aniket/Datasets/",
                 transform = None,
                 ):
        super(InferenceDataset, self).__init__(transform=transform)
        print("data: ", data_dir)
        left_files = sorted(glob(data_dir + '/' + 'left/*.png'))

        for left_name in left_files:
            sample = dict()
            sample['left'] = left_name
            sample['right'] = left_name.replace('left', 'right')

            self.samples.append(sample)

class KITTI15(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/KITTI/',
                 mode='training',
                 transform=None,
                 sequence_length=1,
                 ):
        super(KITTI15, self).__init__(transform=transform, sequence_length=sequence_length)

        # --- 1. Find and Group all files by sequence ---
        # Since KITTI has no sequence organization, we'll simply read all files and put them in individual lists
        left_files = sorted(glob(data_dir + '/' + mode + '/image_2/*_10.png'))
        sequences = defaultdict(list)
        for seq_id in range(len(left_files)):
            all_left_files = [left_files[seq_id]]

            intrinsics_list = [torch.tensor([0.0, 0.0, 0.0, 0.0])]
            poses_list = [torch.eye(4)]
            baseline = torch.tensor(0.10)

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = left_path.replace("image_2", "image_3")
                sample['disp'] = left_path.replace("image_2", "disp_occ_0")
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = baseline

                sequences[seq_id].append(sample)
        
        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

class ETH3DStereo(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/ETH3D',
                 mode='two_view_training',
                 transform=None,
                 sequence_length=1,
                 ):
        super(ETH3DStereo, self).__init__(transform=transform, sequence_length=sequence_length, is_middlebury_eth3d=True)

        # --- 1. Find and Group all files by sequence ---
        seq_root = data_dir + '/' + mode 
        sequences = defaultdict(list)
        for seq in os.listdir(seq_root):
            all_left_files = [os.path.join(seq_root, seq, 'im0.png')]
            
            # Intrinsics don't really matter for single sequence evaluation so putting in 0's for all
            intrinsics_list = []
            with open(os.path.join(seq_root, seq, "calib.txt")) as f:
                intrinsics_list.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))

                # Read the baseline
                for line in f:
                    if line.startswith('baseline='):
                        baseline = torch.tensor(float(line.split('=')[1]))
            
            # Poses are also not required for single sequence so putting in identity
            poses_list = [torch.eye(4)]

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = left_path.replace("im0", "im1")
                sample['disp'] = left_path.replace("im0.png", "disp0GT.pfm").replace("two_view_training", "two_view_training_gt")
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = baseline

                sequences[seq].append(sample)

        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

class MiddleburyEval3(VideoStereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Middlebury/MiddEval3',
                 mode='training',
                 resolution='H',
                 transform=None,
                 sequence_length=1,
                 ):
        super(MiddleburyEval3, self).__init__(transform=transform, sequence_length=sequence_length, is_middlebury_eth3d=True)

        # --- 1. Find and Group all files by sequence ---
        seq_root = data_dir + '/' + mode + resolution
        sequences = defaultdict(list)
        for seq in os.listdir(seq_root):
            all_left_files = [os.path.join(seq_root, seq, 'im0.png')]
            
            # Intrinsics don't really matter for single sequence evaluation so putting in 0's for all
            intrinsics_list = []
            with open(os.path.join(seq_root, seq, "calib.txt")) as f:
                intrinsics_list.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))

                # Read the baseline
                for line in f:
                    if line.startswith('baseline='):
                        baseline = torch.tensor(float(line.split('=')[1]))
            
            # Poses are also not required for single sequence so putting in identity
            poses_list = [torch.eye(4)]

            assert len(all_left_files)==len(intrinsics_list) and len(all_left_files)==len(poses_list), print(len(all_left_files), len(intrinsics_list), len(poses_list)) 

            for left_path, intrinsics, pose in zip(all_left_files, intrinsics_list, poses_list):
                sample = {}

                sample['left'] = left_path
                sample['right'] = left_path.replace("im0", "im1")
                sample['disp'] = left_path.replace("im0.png", "disp0GT.pfm")
                sample['intrinsics'] = intrinsics
                sample['pose'] = pose
                sample['baseline'] = baseline

                sequences[seq].append(sample)

        # --- 2. Populate self.samples with the grouped sequences ---
        # Sort by sequence ID to ensure deterministic order
        for seq_id in sorted(sequences.keys()):
            # Ensure frames within the sequence are sorted correctly (glob should handle this)
            self.samples.append(sequences[seq_id])
            
        # --- 3. Build the list of valid starting points ---
        self.build_valid_starts()

        print(f"Found {len(sequences)} sequences and {len(self.valid_starts)} valid clips of length {self.sequence_length}.")

#endregion =======================================================================================


def build_dataset(args):
    train_transform_list = [
        video_transforms.RandomScale(crop_width=768),
        video_transforms.RandomCrop(384, 768),
        video_transforms.RandomColor(),
        video_transforms.RandomVerticalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    train_transform = video_transforms.Compose(train_transform_list)

    if isinstance(args.stage, str):
        args.stage = [args.stage]  # Ensure stage is a list

    datasets = []
    for stage in args.stage:
        if stage == 'spring':
            datasets.append(SpringDataset(transform=train_transform))
        elif stage == 'flyingthings3d':
            datasets.append(FlyingThings3D(transform=train_transform))
        elif stage == 'monkaa':
            datasets.append(Monkaa(transform=train_transform))
        elif stage == 'driving':
            datasets.append(Driving(transform=train_transform))
        elif stage == 'infinigen':
            datasets.append(InfinigenSV(transform=train_transform))
        else:
            raise ValueError(f"Unknown stage: {stage}")

    if len(datasets) == 0:
        raise ValueError("No valid datasets found for the provided stages.")

    if len(datasets) == 1:
        return datasets[0]  # Return single dataset if only one stage

    return torch.utils.data.ConcatDataset(datasets)  # Concatenate multiple datasets

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