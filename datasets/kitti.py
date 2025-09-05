from glob import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset
import torch

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='/projects/nufr/aniket/Datasets/Stereo_Disp/KITTI',
                 ):
        super(KITTI, self).__init__(aug_params, sparse=True,
                                    )

        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.npy')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.npy')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
 

    def resize_sample(self, img1, img2, disp, valid):
        """Resize using torchvision transforms (tensor-friendly)"""
        import torchvision.transforms.functional as TF
        
        h_target, w_target = self.target_size
        
        # Get original size (handle both numpy and tensor)
        if torch.is_tensor(img1):
            h_orig, w_orig = img1.shape[-2:]
        else:
            h_orig, w_orig = img1.shape[:2]
            # Convert numpy to tensor first
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            if disp is not None:
                disp = torch.from_numpy(disp).float()
            if valid is not None:
                valid = torch.from_numpy(valid).float()
        
        scale_w = w_target / w_orig
        
        # Resize using torchvision
        img1 = TF.resize(img1, (h_target, w_target), interpolation=TF.InterpolationMode.BILINEAR)
        img2 = TF.resize(img2, (h_target, w_target), interpolation=TF.InterpolationMode.BILINEAR)
        
        # Resize disparity
        if disp is not None:
            if len(disp.shape) == 2:
                disp = disp.unsqueeze(0)  # Add channel dimension
            disp = TF.resize(disp, (h_target, w_target), interpolation=TF.InterpolationMode.NEAREST)
            # if disp.shape[0] == 1:
            #     disp = disp.squeeze(0)  # Remove channel dimension
            disp = disp * scale_w  # Scale disparity values
        
        # Resize valid mask
        if valid is not None:
            if len(valid.shape) == 2:
                valid = valid.unsqueeze(0)
            valid = TF.resize(valid, (h_target, w_target), interpolation=TF.InterpolationMode.NEAREST)
 
        
        return img1, img2, disp, valid