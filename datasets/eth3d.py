from glob import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset
import pdb

class ETH3D(FlowDataset):
    def __init__(self, aug_params=None, 
                 split="training",
                 root='/work/nufr/aniket/Datasets/Stereo_Disp/ETH3D',
                 ):
    
        super(ETH3D, self).__init__(aug_params)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)
        # pdb.set_trace()
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disp_list += [ disp ]