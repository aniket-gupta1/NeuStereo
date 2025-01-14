import glob
import os
import os.path as osp
import numpy as np
from .default import FlowDataset

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='datasets/Sintel',
                 dstype='clean',
                 load_occlusion=False,
                 ):
        super(MpiSintel, self).__init__(aug_params,
                                        load_occlusion=load_occlusion,
                                        )

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if load_occlusion:
            occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.npy')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.npy')))

                if load_occlusion:
                    self.occ_list += sorted(glob(osp.join(occlusion_root, scene, '*.png')))
