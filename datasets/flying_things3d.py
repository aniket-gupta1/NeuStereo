import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=False,
                 only_left=True,
                 ):
        super(FlyingThings3D, self).__init__(aug_params)

        img_dir = root
        flow_dir = root

        if only_left:
            cam_list = ['left']
        else: 
            cam_list = ['left', 'right']

        for cam in cam_list:
            for direction in ['into_future', 'into_past']:
                if test_set:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TEST/*/*')))
                else:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                if test_set:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TEST/*/*')))
                else:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.npy')))
                    flows = sorted(glob(osp.join(fdir, '*.npy')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]

        # if test_set and validate_subset:
        if validate_subset:
            num_val_samples = 1024
            all_test_samples = len(self.image_list)  # 7866

            stride = all_test_samples // num_val_samples
            remove = all_test_samples % num_val_samples

            self.image_list = self.image_list[:-remove][::stride]
            self.flow_list = self.flow_list[:-remove][::stride]