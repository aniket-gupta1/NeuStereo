from glob import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset
import pdb

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root='/projects/nufr/aniket/Datasets/Stereo_Disp/FlyingThings3D',
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

        if test_set or validate_subset:
            split = 'TEST'
        else:
            split = 'TRAIN'
        
        left_images = sorted(glob(osp.join(root, dstype, split, '*/*/left/*.npy')))
        right_images = [ im.replace('left', 'right') for im in left_images]
        disparity_images = [ im.replace(dstype, 'disparity').replace('.npy', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disp_list += [ disp ]


    