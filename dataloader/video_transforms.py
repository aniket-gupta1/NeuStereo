from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert a sequence of numpy arrays to a sequence of torch tensors"""
    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample):
        # Input 'sample' is now a dictionary of lists, e.g., {'left': [img1, img2, ...]}
        num_frames = len(sample['left'])
        
        # Process each frame in the sequence
        for i in range(num_frames):
            left_frame = np.transpose(sample['left'][i], (2, 0, 1)) # [3, H, W]
            right_frame = np.transpose(sample['right'][i], (2, 0, 1))

            if self.no_normalize:
                sample['left'][i] = torch.from_numpy(left_frame.copy())
                sample['right'][i] = torch.from_numpy(right_frame.copy())
            else:
                sample['left'][i] = torch.from_numpy(left_frame.copy()) / 255.
                sample['right'][i] = torch.from_numpy(right_frame.copy()) / 255.

            if 'disp' in sample:
                disp_frame = sample['disp'][i] # [H, W]
                sample['disp'][i] = torch.from_numpy(disp_frame.copy())

            if 'intrinsics' in sample:
                sample['intrinsics'][i] = torch.from_numpy(sample['intrinsics'][i].copy())

        return sample

class Normalize(object):
    """Normalize a sequence of images (tensors)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        norm_keys = ['left', 'right']
        num_frames = len(sample['left'])

        for key in norm_keys:
            # Loop through each frame in the sequence
            for i in range(num_frames):
                # sample[key][i] is the tensor for the i-th frame
                # The original logic works per-tensor
                for t, m, s in zip(sample[key][i], self.mean, self.std):
                    t.sub_(m).div_(s)

        return sample

class RandomCrop(object):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        # Use the first frame to determine original size and crop coordinates
        first_left_frame = sample['left'][0]
        ori_height, ori_width = first_left_frame.shape[:2]
        num_frames = len(sample['left'])

        # --- Generate random crop parameters ONCE ---
        top_pad = max(self.img_height - ori_height, 0)
        right_pad = max(self.img_width - ori_width, 0)
        
        # Calculate new dimensions after potential padding
        final_height = ori_height + top_pad
        final_width = ori_width + right_pad

        offset_x = np.random.randint(final_width - self.img_width + 1)
        start_height = 0
        offset_y = np.random.randint(start_height, final_height - self.img_height + 1)

        # --- Apply the SAME crop to all frames in the sequence ---
        for i in range(num_frames):
            # Apply padding if necessary
            if top_pad > 0 or right_pad > 0:
                sample['left'][i] = np.lib.pad(sample['left'][i],
                                               ((top_pad, 0), (0, right_pad), (0, 0)),
                                               mode='edge')
                sample['right'][i] = np.lib.pad(sample['right'][i],
                                                ((top_pad, 0), (0, right_pad), (0, 0)),
                                                mode='edge')
                if 'disp' in sample:
                    sample['disp'][i] = np.lib.pad(sample['disp'][i],
                                                   ((top_pad, 0), (0, right_pad)),
                                                   mode='constant', constant_values=0)

            # Apply the crop
            sample['left'][i] = sample['left'][i][offset_y:offset_y + self.img_height, offset_x:offset_x + self.img_width]
            sample['right'][i] = sample['right'][i][offset_y:offset_y + self.img_height, offset_x:offset_x + self.img_width]
            if 'disp' in sample:
                sample['disp'][i] = sample['disp'][i][offset_y:offset_y + self.img_height, offset_x:offset_x + self.img_width]

            # --- MODIFIED: Update Intrinsics ---
            if 'intrinsics' in sample:
                # [fx, fy, cx, cy]
                
                # 1. Account for padding
                # Pad adds `top_pad` pixels to the top.
                # Pad adds `0` pixels to the left.
                # new_cx = old_cx + 0
                # new_cy = old_cy + top_pad
                
                # 2. Account for crop offset
                # The new origin is (offset_x, offset_y) in the padded space.
                # final_cx = new_cx - offset_x
                # final_cy = new_cy - offset_y
                
                # Combine:
                # cx = cx - offset_x
                sample['intrinsics'][i][2] = sample['intrinsics'][i][2] - offset_x
                # cy = (cy + top_pad) - offset_y
                sample['intrinsics'][i][3] = sample['intrinsics'][i][3] + top_pad - offset_y

        return sample

# Random Coloring
class RandomColor(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.transforms = [
            RandomContrast(asymmetric_color_aug=self.asymmetric_color_aug),
            RandomGamma(asymmetric_color_aug=self.asymmetric_color_aug),
            RandomBrightness(asymmetric_color_aug=self.asymmetric_color_aug),
            RandomHue(asymmetric_color_aug=self.asymmetric_color_aug),
            RandomSaturation(asymmetric_color_aug=self.asymmetric_color_aug)
        ]

    def __call__(self, sample):
        num_frames = len(sample['left'])
        
        # --- Convert all frames to PIL Images first ---
        for i in range(num_frames):
            sample['left'][i] = Image.fromarray(sample['left'][i].astype('uint8'))
            sample['right'][i] = Image.fromarray(sample['right'][i].astype('uint8'))

        # --- Decide on the augmentation sequence ONCE ---
        if np.random.random() < 0.5:
            # A single transform
            transform_to_apply = random.choice(self.transforms)
            # Generate random parameters within the chosen transform
            transform_to_apply.generate_params()
            
            # Apply the same transform with the same params to all frames
            for i in range(num_frames):
                sample['left'][i], sample['right'][i] = transform_to_apply.apply(
                    sample['left'][i], sample['right'][i]
                )

        else:
            # Combination of transforms in a random order
            random.shuffle(self.transforms)
            for t in self.transforms:
                t.generate_params() # Generate params for this transform
                for i in range(num_frames):
                    sample['left'][i], sample['right'][i] = t.apply(
                        sample['left'][i], sample['right'][i]
                    )

        # --- Convert all frames back to Numpy Arrays ---
        for i in range(num_frames):
            sample['left'][i] = np.array(sample['left'][i]).astype(np.float32)
            sample['right'][i] = np.array(sample['right'][i]).astype(np.float32)

        return sample

class RandomContrast(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.factor1 = 1.0
        self.factor2 = 1.0

    def generate_params(self):
        """Generate random parameters once per call."""
        if np.random.random() < 0.5:
            self.factor1 = np.random.uniform(0.8, 1.2)
            self.factor2 = self.factor1
            if self.asymmetric_color_aug and np.random.random() < 0.5:
                self.factor2 = np.random.uniform(0.8, 1.2)
    
    def apply(self, left_img, right_img):
        """Apply the transform with pre-generated parameters."""
        left_aug = F.adjust_contrast(left_img, self.factor1)
        right_aug = F.adjust_contrast(right_img, self.factor2)
        return left_aug, right_aug

class RandomGamma(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.factor1 = 1.0
        self.factor2 = 1.0
    
    def generate_params(self):
        """Generate random parameters once per call."""
        if np.random.random() < 0.5:
            self.factor1 = np.random.uniform(0.7, 1.5)
            self.factor2 = self.factor1
            if self.asymmetric_color_aug and np.random.random() < 0.5:
                self.factor2 = np.random.uniform(0.7, 1.5)
    
    def apply(self, left_img, right_img):
        """Apply the transform with pre-generated parameters."""
        left_aug = F.adjust_gamma(left_img, self.factor1)
        right_aug = F.adjust_gamma(right_img, self.factor2)
        return left_aug, right_aug

class RandomBrightness(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.factor1 = 1.0
        self.factor2 = 1.0
    
    def generate_params(self):
        """Generate random parameters once per call."""
        if np.random.random() < 0.5:
            self.factor1 = np.random.uniform(0.5, 2.0)
            self.factor2 = self.factor1
            if self.asymmetric_color_aug and np.random.random() < 0.5:
                self.factor2 = np.random.uniform(0.5, 2.0)
    
    def apply(self, left_img, right_img):
        """Apply the transform with pre-generated parameters."""
        left_aug = F.adjust_brightness(left_img, self.factor1)
        right_aug = F.adjust_brightness(right_img, self.factor2)
        return left_aug, right_aug

class RandomHue(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.factor1 = 0.0
        self.factor2 = 0.0
    
    def generate_params(self):
        """Generate random parameters once per call."""
        if np.random.random() < 0.5:
            self.factor1 = np.random.uniform(-0.1, 0.1)
            self.factor2 = self.factor1
            if self.asymmetric_color_aug and np.random.random() < 0.5:
                self.factor2 = np.random.uniform(-0.1, 0.1)
    
    def apply(self, left_img, right_img):
        """Apply the transform with pre-generated parameters."""
        left_aug = F.adjust_hue(left_img, self.factor1)
        right_aug = F.adjust_hue(right_img, self.factor2)
        return left_aug, right_aug

class RandomSaturation(object):
    def __init__(self, asymmetric_color_aug=True):
        self.asymmetric_color_aug = asymmetric_color_aug
        self.factor1 = 1.0
        self.factor2 = 1.0
    
    def generate_params(self):
        """Generate random parameters once per call."""
        if np.random.random() < 0.5:
            self.factor1 = np.random.uniform(0.8, 1.2)
            self.factor2 = self.factor1
            if self.asymmetric_color_aug and np.random.random() < 0.5:
                self.factor2 = np.random.uniform(0.8, 1.2)
    
    def apply(self, left_img, right_img):
        """Apply the transform with pre-generated parameters."""
        left_aug = F.adjust_saturation(left_img, self.factor1)
        right_aug = F.adjust_saturation(right_img, self.factor2)
        return left_aug, right_aug


class RandomVerticalFlip(object):
    """Randomly vertically flips the entire sequence."""

    def __call__(self, sample):
        # --- Decide to flip ONCE for the whole sequence ---
        if np.random.random() < 0.5:
            num_frames = len(sample['left'])
            
            # --- Apply the SAME flip to all frames ---
            for i in range(num_frames):
                # --- MODIFIED: Get height *before* flipping ---
                h = sample['left'][i].shape[0]

                sample['left'][i] = np.copy(np.flipud(sample['left'][i]))
                sample['right'][i] = np.copy(np.flipud(sample['right'][i]))

                if 'disp' in sample:
                    sample['disp'][i] = np.copy(np.flipud(sample['disp'][i]))
                
                # --- MODIFIED: Update Intrinsics ---
                if 'intrinsics' in sample:
                    # [fx, fy, cx, cy]
                    # new_cy = (h - 1) - old_cy
                    sample['intrinsics'][i][3] = (h - 1) - sample['intrinsics'][i][3]

        return sample
    

class RandomScale(object):
    def __init__(self,
                 min_scale=-0.4,
                 max_scale=0.4,
                 crop_width=512,
                 nearest_interp=False,
                 ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_width = crop_width
        self.nearest_interp = nearest_interp

    def __call__(self, sample):
        # --- Decide to scale and generate scale factor ONCE ---
        if np.random.rand() < 0.5:
            # Use the first frame to get dimensions
            h, w = sample['left'][0].shape[:2]
            num_frames = len(sample['left'])

            scale_x = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = np.clip(scale_x, self.crop_width / float(w), None)

            # --- Apply the SAME scaling to all frames ---
            for i in range(num_frames):
                # only random scale x axis
                sample['left'][i] = cv2.resize(sample['left'][i], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)
                sample['right'][i] = cv2.resize(sample['right'][i], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)
                
                if 'disp' in sample:                    
                    sample['disp'][i] = cv2.resize(
                        sample['disp'][i], None, fx=scale_x, fy=1.,
                        interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
                    ) * scale_x

                # --- MODIFIED: Update Intrinsics ---
                if 'intrinsics' in sample:
                    # [fx, fy, cx, cy]
                    # Since only x-axis is scaled (scale_y = 1.0):
                    # fx scales with scale_x
                    # cx scales with scale_x
                    # fy and cy are unchanged
                    sample['intrinsics'][i][0] = sample['intrinsics'][i][0] * scale_x # fx
                    sample['intrinsics'][i][2] = sample['intrinsics'][i][2] * scale_x # cx

        return sample