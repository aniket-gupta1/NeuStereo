from glob import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset
import pdb

class Driving(FlowDataset):
    def __init__(self, aug_params=None,
                 root='/projects/nufr/aniket/Datasets/Stereo_Disp/Driving',
                 dstype='frames_cleanpass',
                 ):
        super(Driving, self).__init__(aug_params)

        left_images = sorted( glob(osp.join(root, dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disp_list += [ disp ]


    
