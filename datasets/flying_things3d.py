from glob import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset
import pdb

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root='/work/nufr/aniket/Datasets/Stereo_Disp/FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=False,
                 only_left=False,
                 ):
        super(FlyingThings3D, self).__init__(aug_params)

        img_dir = root
        flow_dir = root

        if only_left:
            cam_list = ['left']
        else: 
            cam_list = ['left', 'right']

        if test_set:
            split = 'TEST'
        else:
            split = 'TRAIN'

        # for cam in cam_list:
        #     if test_set:
        #         image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TEST/*/*')))
        #     else:
        #         image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TRAIN/*/*')))
        #     image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

        #     if test_set:
        #         disp_dirs = sorted(glob(osp.join(flow_dir, 'disparity/TEST/*/*')))
        #     else:
        #         disp_dirs = sorted(glob(osp.join(flow_dir, 'disparity/TRAIN/*/*')))
        #     disp_dirs = sorted([osp.join(f, cam) for f in disp_dirs])
            

        #     for idir, fdir in zip(image_dirs, disp_dirs):
        #         images = sorted(glob(osp.join(idir, '*.npy')))
        #         disp = sorted(glob(osp.join(fdir, '*.pfm')))
        #         for i in range(len(flows) - 1):
        #             if direction == 'into_future':
        #                 self.image_list += [[images[i], images[i + 1]]]
        #                 self.flow_list += [flows[i]]
        #             elif direction == 'into_past':
        #                 self.image_list += [[images[i + 1], images[i]]]
        #                 self.flow_list += [flows[i + 1]]
        
        left_images = sorted(glob(osp.join(root, dstype, split, '*/*/left/*.npy')))
        right_images = [ im.replace('left', 'right') for im in left_images]
        disparity_images = [ im.replace(dstype, 'disparity').replace('.npy', '.pfm') for im in left_images]

        # # Choose a random subset of 400 images for validation
        # state = np.random.get_state()
        # np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:400])
        # np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disp_list += [ disp ]

        

        # # if test_set and validate_subset:
        # if validate_subset:
        #     num_val_samples = 1024
        #     all_test_samples = len(self.image_list)  # 7866

        #     stride = all_test_samples // num_val_samples
        #     remove = all_test_samples % num_val_samples

        #     self.image_list = self.image_list[:-remove][::stride]
        #     self.flow_list = self.flow_list[:-remove][::stride]