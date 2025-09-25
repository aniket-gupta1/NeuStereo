import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2
import hashlib
import json
import time
import logging
import os.path as osp
from utils.file_io import read_img, read_disp

from . import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataset(Dataset):
    def __init__(self,
                 transform=None,
                 is_vkitti2=False,
                 is_sintel=False,
                 is_middlebury_eth3d=False,
                 is_tartanair=False,
                 is_instereo2k=False,
                 is_crestereo=False,
                 is_fallingthings=False,
                 is_raw_disp_png=False,
                 is_FSD=False,
                 half_resolution=False,
                 ):

        super(StereoDataset, self).__init__()

        self.transform = transform
        self.save_filename = False

        self.is_vkitti2 = is_vkitti2
        self.is_sintel = is_sintel
        self.is_middlebury_eth3d = is_middlebury_eth3d
        self.is_tartanair = is_tartanair
        self.is_instereo2k = is_instereo2k
        self.is_crestereo = is_crestereo
        self.is_fallingthings = is_fallingthings
        self.half_resolution = half_resolution
        self.is_raw_disp_png = is_raw_disp_png
        self.is_FSD = is_FSD

        self.samples = []

    def __getitem__(self, index):
        sample = {}

        # file path
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        if 'disp' in sample_path and sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'],
                                       vkitti2=self.is_vkitti2,
                                       sintel=self.is_sintel,
                                       tartanair=self.is_tartanair,
                                       instereo2k=self.is_instereo2k,
                                       fallingthings=self.is_fallingthings,
                                       crestereo=self.is_crestereo,
                                       FSD=self.is_FSD,
                                       raw_disp_png=self.is_raw_disp_png,
                                       )  # [H, W]

            # for middlebury and eth3d datasets, invalid is denoted as inf
            if self.is_middlebury_eth3d or self.is_crestereo:
                sample['disp'][sample['disp'] == np.inf] = 0

        if self.half_resolution:
            sample['left'] = cv2.resize(sample['left'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            sample['right'] = cv2.resize(sample['right'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            sample['disp'] = cv2.resize(sample['disp'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) * 0.5

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples

        return self

class FoundationStereo(StereoDataset):
    def __init__(self,
                 data_dir='/projects/NEUFR/data/FSD',
                 test_set=False,
                 validate_subset=False,
                 only_left=False,
                 split_folders='all',  # List of split folders or 'all' or single string
                 max_scenes_per_split=None,  # Limit scenes per split to manage memory
                 use_cache=True,
                 transform=None,
                 ):
        super(FoundationStereo, self).__init__(transform=transform, is_FSD=True)

        # samples: train: 1108890, test: 0

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
            if success:
                return

        # logging.info("Building dataset from scratch (slow)...")

        # Handle split_folders parameter
        if split_folders is None:
            split_folders = ['0000000']  # Default to first split
        elif split_folders == 'all':
            # Use all available splits
            split_folders = self.list_available_splits(data_dir)
            if not split_folders:
                raise ValueError(f"No split folders found in {data_dir}")
        elif isinstance(split_folders, str):
            # Single split folder provided as string
            split_folders = [split_folders]

        # logging.info(f"Using split folders: {split_folders}")

        all_left_images = []
        all_right_images = []
        all_disparity_images = []
        
        # Iterate through each split folder
        for split_folder in split_folders:
            logging.info(f"\nProcessing split: {split_folder}")
            
            # Get all scene directories for this split
            scene_pattern = osp.join(data_dir, split_folder, '*', 'dataset', 'data')
            scene_dirs = sorted(glob(scene_pattern))
            
            if not scene_dirs:
                logging.info(f"  Warning: No scene directories found in {split_folder}")
                continue
            
            # Limit scenes per split if specified
            if max_scenes_per_split:
                scene_dirs = scene_dirs[:max_scenes_per_split]
                logging.info(f"  Limited to {len(scene_dirs)} scenes (max_scenes_per_split={max_scenes_per_split})")
            
            logging.info(f"Found {len(scene_dirs)} scenes")
            
            split_left_images = []
            split_right_images = []
            split_disparity_images = []
            
            # Collect all images from all scenes in this split
            for scene_dir in scene_dirs:
                scene_name = osp.basename(osp.dirname(osp.dirname(scene_dir)))
                
                # Get left RGB images
                left_rgb_dir = osp.join(scene_dir, 'left', 'rgb')
                left_rgb_pattern = osp.join(left_rgb_dir, '*')
                scene_left_images = sorted(glob(left_rgb_pattern))
                
                # Get corresponding right RGB images and disparity images
                for left_img in scene_left_images:
                    img_name = osp.basename(left_img)
                    
                    # Right RGB image (same filename in right/rgb/)
                    right_img = osp.join(scene_dir, 'right', 'rgb', img_name)
                    
                    # Disparity image (same filename in left/disparity/)
                    disp_base = osp.splitext(img_name)[0]
                    
                    # Try common disparity extensions
                    disp_extensions = ['.pfm', '.png', '.exr', '.tiff', '.tif']
                    disp_img = None
                    
                    for ext in disp_extensions:
                        disp_candidate = osp.join(scene_dir, 'left', 'disparity', disp_base + ext)
                        if osp.exists(disp_candidate):
                            disp_img = disp_candidate
                            break
                    
                    # Only add if all three files exist
                    if osp.exists(left_img) and osp.exists(right_img) and disp_img and osp.exists(disp_img):
                        split_left_images.append(left_img)
                        split_right_images.append(right_img)
                        split_disparity_images.append(disp_img)
            
            logging.info(f"Collected {len(split_left_images)} image triplets from {split_folder}")
            
            # Add to overall lists
            all_left_images.extend(split_left_images)
            all_right_images.extend(split_right_images)
            all_disparity_images.extend(split_disparity_images)

        logging.info(f"\nTotal images collected: {len(all_left_images)} triplets from {len(split_folders)} splits")
        
        if len(all_left_images) == 0:
            raise ValueError("No valid image triplets found!")
        
        # Put all the data in the main samples list
        for left, right, disparity in zip(all_left_images, all_right_images, all_disparity_images):
            sample = {
                'left': left,
                'right': right,
                'disp': disparity
            }

            self.samples.append(sample)
        
        # Save cache
        if use_cache:
            logging.info(f"Saving dataset to cache: {cache_file}")
            cache_data = {
                'image_list': self.image_list,
                'disp_list': self.disp_list,
                'created_time': time.time(),
                'parameters': cache_params,
                'total_samples': len(self.image_list)
            }
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logging.info("Cache saved successfully!")
            except Exception as e:
                logging.info(f"Failed to save cache: {e}")

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
                    self.samples.append(sample)

                
                # logging.info(f"Loaded {len(cached_data['image_list'])} samples from cache")
                # logging.info(f"Cache created: {time.ctime(cached_data.get('created_time', 0))}")

                return True
            else:
                logging.info("Cache parameters don't match, rebuilding...")

        except Exception as e:
            logging.info(f"Cache loading failed: {e}. Rebuilding dataset...")
            return False
    
    @staticmethod
    def list_available_splits(root):
        """List all available split folders (0000000, 0000001, etc.)"""
        splits = []
        if osp.exists(root):
            for item in os.listdir(root):
                item_path = osp.join(root, item)
                if osp.isdir(item_path) and item.isdigit():
                    splits.append(item)
        return sorted(splits)

class InferenceDataset(StereoDataset):
    def __init__(self,
                 data_dir = "/projects/nufr/aniket/Datasets/",
                 transform = None,
                 ):
        super(InferenceDataset, self).__init__(transform=transform)

        left_files = sorted(glob(data_dir + '/' + 'left/*.jpg'))

        for left_name in left_files:
            sample = dict()
            sample['left'] = left_name
            sample['right'] = left_name.replace('left', 'right')

            self.samples.append(sample)
    


class FlyingThings3D(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/FlyingThings3D',
                 mode='TRAIN',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(FlyingThings3D, self).__init__(transform=transform,
                                             )

        # samples: train: 22390, test: 4370
        left_files = sorted(glob(data_dir + '/' + split + '/' + mode + '/*/*/left/*.npy'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class Monkaa(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Monkaa',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(Monkaa, self).__init__(transform=transform)

        # samples: 8664
        left_files = sorted(glob(data_dir + '/' + split + '/*/left/*.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class Driving(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Driving',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(Driving, self).__init__(transform=transform)

        # samples: 4400
        left_files = sorted(glob(data_dir + '/' + split + '/*/*/*/left/*.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class KITTI15(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/KITTI/',
                 mode='training',
                 transform=None,
                 save_filename=False,
                 ):
        super(KITTI15, self).__init__(transform=transform)

        assert mode in ['training', 'testing']

        self.save_filename = save_filename

        # samples: train: 200
        left_files = sorted(glob(data_dir + '/' + mode + '/image_2/*_10.png'))

        if mode == 'testing':
            self.save_filename = True

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('image_2', 'image_3')

            if mode != 'testing':
                sample['disp'] = left_name.replace('image_2', 'disp_occ_0')

            if mode == 'testing' or self.save_filename:
                # save filename
                sample['left_name'] = os.path.basename(left_name)

            self.samples.append(sample)


class KITTI12(StereoDataset):
    def __init__(self,
                 data_dir='datasets/KITTI/stereo/kitti_2012',
                 mode='training',
                 transform=None,
                 ):
        super(KITTI12, self).__init__(transform=transform)

        assert mode in ['training', 'testing']

        if mode == 'testing':
            self.save_filename = True

        # samples: train: 195
        left_files = sorted(glob(data_dir + '/' + mode + '/colored_0/*_10.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/colored_0/', '/colored_1/')

            if mode != 'testing':
                sample['disp'] = left_name.replace('/colored_0/', '/disp_occ/')

            if mode == 'testing':
                # save filename
                sample['left_name'] = os.path.basename(left_name)

            self.samples.append(sample)


class VKITTI2(StereoDataset):
    def __init__(self,
                 data_dir='datasets/VKITTI2',
                 transform=None,
                 ):
        super(VKITTI2, self).__init__(transform=transform,
                                      is_vkitti2=True,
                                      )

        # total: 21260
        left_files = sorted(glob(data_dir + '/Scene*/*/frames/rgb/Camera_0/rgb*.jpg'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/Camera_0/', '/Camera_1/')
            sample['disp'] = left_name.replace('/rgb/', '/depth/').replace('rgb_', 'depth_')[:-3] + 'png'

            self.samples.append(sample)


class DrivingStereo(StereoDataset):
    def __init__(self,
                 data_dir='datasets/DrivingStereo',
                 transform=None,
                 ):
        super(DrivingStereo, self).__init__(transform=transform)

        # total: 174437
        left_files = sorted(glob(data_dir + '/train-left-image/*/*.jpg'))
        right_files = sorted(glob(data_dir + '/train-right-image/*/*.jpg'))
        disp_files = sorted(glob(data_dir + '/train-disparity-map/*/*.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class SintelStereo(StereoDataset):
    def __init__(self,
                 data_dir='datasets/SintelStereo',
                 split='clean',
                 transform=None,
                 save_filename=False,
                 ):
        super(SintelStereo, self).__init__(transform=transform, is_sintel=True)

        self.save_filename = save_filename

        assert split in ['clean', 'final']

        # total: clean & final each 1064
        left_files = sorted(glob(data_dir + '/training/' + split + '_left/*/*.png'))
        right_files = sorted(glob(data_dir + '/training/' + split + '_right/*/*.png'))
        disp_files = sorted(glob(data_dir + '/training/disparities/*/*.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            if self.save_filename:
                sample['left_name'] = left_files[i]

            self.samples.append(sample)


class ETH3DStereo(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/ETH3D',
                 mode='train',
                 transform=None,
                 save_filename=False,
                 ):
        super(ETH3DStereo, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        if mode == 'train':
            left_files = sorted(glob(data_dir + '/two_view_training/*/im0.png'))
            right_files = sorted(glob(data_dir + '/two_view_training/*/im1.png'))
        else:
            left_files = sorted(glob(data_dir + '/two_view_test/*/im0.png'))
            right_files = sorted(glob(data_dir + '/two_view_test/*/im1.png'))

        if mode == 'train':
            disp_files = sorted(glob(data_dir + '/two_view_training_gt/*/disp0GT.pfm'))
            assert len(left_files) == len(right_files) == len(disp_files)
        else:
            assert len(left_files) == len(right_files)

        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]

            if mode == 'train':
                sample['disp'] = disp_files[i]

            if save_filename:
                sample['left_name'] = left_files[i]

            self.samples.append(sample)


class MiddleburyEval3(StereoDataset):
    def __init__(self,
                 data_dir='/projects/nufr/aniket/Datasets/Stereo_Disp/Middlebury/MiddEval3',
                 mode='training',
                 resolution='H',
                 transform=None,
                 save_filename=False,
                 ):
        super(MiddleburyEval3, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        assert mode in ['training', 'test']
        assert resolution in ['Q', 'H', 'F']

        left_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/im0.png'))
        right_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/im1.png'))

        if mode == 'training':
            disp_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/disp0GT.pfm'))
            assert len(left_files) == len(right_files) == len(disp_files)
        else:
            assert len(left_files) == len(right_files)

        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]

            if mode == 'training':
                sample['disp'] = disp_files[i]

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)


class Middlebury20052006(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2005',
                 transform=None,
                 save_filename=False,
                 ):
        super(Middlebury20052006, self).__init__(transform=transform, is_raw_disp_png=True)

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            # Middlebury/2005/Art
            sample = dict()

            sample['left'] = os.path.join(data_dir, curr_dir, 'view1.png')
            sample['right'] = os.path.join(data_dir, curr_dir, 'view5.png')
            sample['disp'] = os.path.join(data_dir, curr_dir, 'disp1.png')

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)

            # same disp for different images
            gt_disp = os.path.join(data_dir, curr_dir, 'disp1.png')

            # also include different illuminations
            for illum in ['Illum1', 'Illum2', 'Illum3']:
                for exp in ['Exp0', 'Exp1', 'Exp2']:
                    # Middlebury/2005/Art/Illum1/Exp0/
                    sample = dict()

                    sample['left'] = os.path.join(data_dir, curr_dir, illum, exp, 'view1.png')
                    sample['right'] = os.path.join(data_dir, curr_dir, illum, exp, 'view5.png')
                    sample['disp'] = gt_disp

                    if save_filename:
                        sample['left_name'] = sample['left']

                    self.samples.append(sample)


class Middlebury2014(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2014',
                 transform=None,
                 save_filename=False,
                 half_resolution=True,
                 ):
        super(Middlebury2014, self).__init__(transform=transform, is_middlebury_eth3d=True,
                                             half_resolution=half_resolution,
                                             )

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            for data_type in ['', 'E', 'L']:
                sample = dict()

                sample['left'] = os.path.join(data_dir, curr_dir, 'im0.png')
                sample['right'] = os.path.join(data_dir, curr_dir, 'im1' + '%s.png' % data_type)
                sample['disp'] = os.path.join(data_dir, curr_dir, 'disp0.pfm')

                if save_filename:
                    sample['left_name'] = sample['left']

                self.samples.append(sample)


class Middlebury2021(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2021/data',
                 transform=None,
                 save_filename=False,
                 ):
        super(Middlebury2021, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            # Middlebury/2021/artroom1
            sample = dict()

            sample['left'] = os.path.join(data_dir, curr_dir, 'im0.png')
            sample['right'] = os.path.join(data_dir, curr_dir, 'im1.png')
            sample['disp'] = os.path.join(data_dir, curr_dir, 'disp0.pfm')

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)

            # same disp for different images
            gt_disp = os.path.join(data_dir, curr_dir, 'disp0.pfm')

            # Middlebury/2021/data/artroom1/ambient/F0
            curr_img_dir = os.path.join(data_dir, curr_dir, 'ambient')

            # also include different illuminations
            # for data_type in ['F0', 'L0', 'L1', 'L2', 'T0']:
            # only use 'L0' lighting since others are too challenging
            for data_type in ['L0']:
                img0s = sorted(glob(curr_img_dir + '/' + data_type + '/im0e*.png'))

                for img0 in img0s:
                    sample = dict()

                    sample['left'] = img0
                    sample['right'] = img0.replace('/im0', '/im1')
                    assert os.path.isfile(sample['right'])

                    sample['disp'] = gt_disp

                    if save_filename:
                        sample['left_name'] = sample['left']

                    self.samples.append(sample)


class CREStereoDataset(StereoDataset):
    def __init__(self,
                 data_dir='datasets/CREStereo/stereo_trainset/crestereo',
                 transform=None,
                 ):
        super(CREStereoDataset, self).__init__(transform=transform, is_crestereo=True)

        left_files = sorted(glob(data_dir + '/*/*_left.jpg'))
        right_files = sorted(glob(data_dir + '/*/*_right.jpg'))
        disp_files = sorted(glob(data_dir + '/*/*_left.disp.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class TartanAir(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Tartanair',
                 transform=None,
                 ):
        super(TartanAir, self).__init__(transform=transform, is_tartanair=True)

        left_files = sorted(glob(data_dir + '/*/*/*/*/image_left/*.png'))
        right_files = sorted(glob(data_dir + '/*/*/*/*/image_right/*.png'))
        disp_files = sorted(glob(data_dir + '/*/*/*/*/depth_left/*.npy'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class CARLA(StereoDataset):
    def __init__(self,
                 data_dir='datasets/HR-VS-Stereo/carla-highres/trainingF',
                 transform=None,
                 ):
        super(CARLA, self).__init__(transform=transform, is_middlebury_eth3d=True,
                                    half_resolution=True)

        left_files = sorted(glob(data_dir + '/*/im0.png'))
        right_files = sorted(glob(data_dir + '/*/im1.png'))
        disp_files = sorted(glob(data_dir + '/*/disp0GT.pfm'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class InStereo2K(StereoDataset):
    def __init__(self,
                 data_dir='datasets/InStereo2K',
                 transform=None,
                 ):
        super(InStereo2K, self).__init__(transform=transform, is_instereo2k=True)

        # merge train and test
        left_files = sorted(glob(data_dir + '/train/*/*/left.png') + glob(data_dir + '/test/*/left.png'))
        right_files = sorted(glob(data_dir + '/train/*/*/right.png') + glob(data_dir + '/test/*/right.png'))
        disp_files = sorted(glob(data_dir + '/train/*/*/left_disp.png') + glob(data_dir + '/test/*/left_disp.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class FallingThings(StereoDataset):
    def __init__(self,
                 data_dir='datasets/FallingThings',
                 transform=None,
                 ):
        super(FallingThings, self).__init__(transform=transform, is_fallingthings=True)

        # merge train and test
        left_files = sorted(glob(data_dir + '/*/*/*left.jpg'))
        right_files = sorted(glob(data_dir + '/*/*/*right.jpg'))
        disp_files = sorted(glob(data_dir + '/*/*/*left.depth.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


def build_dataset(args):
    if args.stage == 'sceneflow':
        train_transform_list = [transforms.RandomScale(crop_width=768),
                                transforms.RandomCrop(384, 768),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        things = FlyingThings3D(transform=train_transform)
        monkaa = Monkaa(transform=train_transform)
        driving = Driving(transform=train_transform)

        train_dataset = things + monkaa + driving

        return train_dataset
    
    elif args.stage == "FSD":
        train_transform_list = [transforms.RandomScale(crop_width=768),
                                transforms.RandomCrop(384, 768),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        foundation_stereo = FoundationStereo(transform=train_transform)
        train_dataset = foundation_stereo

        return train_dataset

    elif args.stage == 'vkitti2':
        train_transform_list = [transforms.RandomScale(crop_width=args.img_width),
                                transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        train_dataset = VKITTI2(transform=train_transform)

        return train_dataset

    elif args.stage == 'kitti15mix':
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # kitti15: 200
        kitti15 = KITTI15(transform=train_transform,
                          )

        # kitti12: 195
        kitti12 = KITTI12(transform=train_transform,
                          )

        train_dataset = 200 * kitti15 + 100 * kitti12

        return train_dataset

    elif args.stage == 'eth3d':
        # dense gt with random resize augmentation
        train_transform_list = [
            transforms.RandomScale(max_scale=0.4,
                                   crop_width=args.img_width),
            transforms.RandomCrop(args.img_height, args.img_width),
            transforms.RandomColor(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

        train_transform = transforms.Compose(train_transform_list)

        # tartanair: 306637
        tartanair = TartanAir(transform=train_transform)

        # sceneflow: 35454
        things = FlyingThings3D(transform=train_transform)
        monkaa = Monkaa(transform=train_transform)
        driving = Driving(transform=train_transform)

        # sintel: 2128
        sintel = SintelStereo(transform=train_transform)

        # crestereo: 200000
        crestereo = CREStereoDataset(transform=train_transform)

        # sparse gt without random scaling
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # eth3d: 27
        eth3d = ETH3DStereo(transform=train_transform)

        # instereo2K: 2010
        instereo2k = InStereo2K(transform=train_transform)

        train_dataset = tartanair + things + monkaa + driving + 50 * sintel + 1000 * eth3d + \
                        100 * instereo2k + 2 * crestereo

        return train_dataset

    elif args.stage == 'eth3d_ft':
        # dense gt with random resize augmentation
        train_transform_list = [
            transforms.RandomScale(max_scale=0.4,
                                   crop_width=args.img_width),
            transforms.RandomCrop(args.img_height, args.img_width),
            transforms.RandomColor(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

        train_transform = transforms.Compose(train_transform_list)

        # crestereo: 200000
        crestereo = CREStereoDataset(transform=train_transform)

        # sparse gt without random scaling
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # eth3d: 27
        eth3d = ETH3DStereo(transform=train_transform)

        # instereo2K: 2010
        instereo2k = InStereo2K(transform=train_transform)

        train_dataset = 1000 * eth3d + 10 * instereo2k + crestereo

        return train_dataset

    elif args.stage == 'middlebury':
        # low res dataset dense gt with random resize augmentation
        # with random rotate shift right image
        train_transform_list = [transforms.RandomScale(min_scale=0,
                                                       max_scale=1.0,
                                                       crop_width=args.img_width),
                                transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomRotateShiftRight(),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # tartanair: 306637
        tartanair = TartanAir(transform=train_transform)

        # sceneflow: 35454
        things = FlyingThings3D(transform=train_transform)
        monkaa = Monkaa(transform=train_transform)
        driving = Driving(transform=train_transform)

        # fallingthings: 31500
        fallingthings = FallingThings(transform=train_transform)

        # high res data transform
        train_transform_list = [transforms.RandomScale(min_scale=-0.2,
                                                       max_scale=0.4,
                                                       crop_width=args.img_width,
                                                       nearest_interp=True,
                                                       ),
                                transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomRotateShiftRight(),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # calar HR-VS: 780
        carla = CARLA(transform=train_transform)

        # crestereo: 200000
        crestereo = CREStereoDataset(transform=train_transform)

        # instereo2K: 2010
        instereo2k = InStereo2K(transform=train_transform)

        # middlebury 2015: 60
        mb2005 = Middlebury20052006(transform=train_transform)
        # middlebury 2016: 210
        mb2006 = Middlebury20052006(data_dir='datasets/Middlebury/2006',
                                    transform=train_transform
                                    )

        # middlebury 2014: 138, use half resolution
        mb2014 = Middlebury2014(half_resolution=True,
                                transform=train_transform)

        # middlebury 2021: 115
        mb2021 = Middlebury2021(transform=train_transform)

        # middlebury eval3: 15
        mbeval3 = MiddleburyEval3(transform=train_transform)

        train_dataset = tartanair + things + monkaa + driving + \
                        fallingthings + 50 * instereo2k + 50 * carla + crestereo + \
                        200 * mb2005 + 200 * mb2006 + 200 * mb2014 + 200 * mb2021 + 200 * mbeval3

        return train_dataset

    elif args.stage == 'middlebury_ft':
        # finetune on high resolution dataset only
        # high res data transform
        train_transform_list = [transforms.RandomScale(min_scale=-0.6,
                                                       max_scale=0.2,
                                                       crop_width=args.img_width,
                                                       nearest_interp=True,
                                                       ),
                                transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomRotateShiftRight(),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # calar HR-VS: 780
        carla = CARLA(transform=train_transform)

        # crestereo: 200000
        crestereo = CREStereoDataset(transform=train_transform)

        # instereo2K: 2010
        instereo2k = InStereo2K(transform=train_transform)

        # middlebury 2015: 60
        mb2005 = Middlebury20052006(transform=train_transform)
        # middlebury 2016: 210
        mb2006 = Middlebury20052006(data_dir='datasets/Middlebury/2006',
                                    transform=train_transform
                                    )

        # middlebury 2014: 138, use half resolution
        mb2014 = Middlebury2014(half_resolution=False,
                                transform=train_transform)

        mb2014_half = Middlebury2014(half_resolution=True,
                                     transform=train_transform)

        # middlebury 2021: 115
        mb2021 = Middlebury2021(transform=train_transform)

        # middlebury eval3: 15
        mbeval3 = MiddleburyEval3(transform=train_transform,
                                  resolution='F',
                                  )

        # middlebury eval3 half: 15
        mbeval3_half = MiddleburyEval3(transform=train_transform,
                                       resolution='H',
                                       )

        # original size 540x960, resize larger
        train_transform_list = [
            transforms.Resize(scale_x=1.2, scale_y=1.5,
                              nearest_interp=False),
            transforms.RandomScale(crop_width=args.img_width,
                                   nearest_interp=True,
                                   ),
            transforms.RandomCrop(args.img_height, args.img_width),
            transforms.RandomRotateShiftRight(),
            transforms.RandomColor(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

        train_transform = transforms.Compose(train_transform_list)

        # fallingthings: 31500, original size 540x960, resize larger
        fallingthings = FallingThings(transform=train_transform)

        train_dataset = crestereo + 50 * instereo2k + 50 * carla + \
                        200 * mb2005 + 200 * mb2006 + 200 * mb2014 + \
                        200 * mb2021 + 200 * mbeval3 + 200 * mb2014_half + 200 * mbeval3_half + \
                        10 * fallingthings

        return train_dataset

    else:
        raise NotImplementedError