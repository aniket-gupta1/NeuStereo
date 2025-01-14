from .flying_chairs import FlyingChairs
from .flying_things3d import FlyingThings3D
from .mpi_sintel import MpiSintel
from .kitti import KITTI
import torch
from torch.utils.data import Dataset, DataLoader

class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.cumsum_lens = [sum(self.lens[:i+1]) for i in range(len(self.lens))]
    
    def __len__(self):
        return sum(self.lens)
    
    def __getitem__(self, index):
        dataset_index = next(i for i, v in enumerate(self.cumsum_lens) if v > index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumsum_lens[dataset_index-1]

        
        data = self.datasets[dataset_index][sample_index]
        return data

def build_dataset(config, stage, split='train'):
    if stage == 'chairs':
        aug_params = {'crop_size': (384, 512), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}

        train_dataset = FlyingChairs(aug_params, split='training')

    elif stage == 'things':
        aug_params = {'crop_size': (384, 768), 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif stage == 'sintel':
        crop_size = (368, 768)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        aug_params = {'crop_size': crop_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        kitti = KITTI(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}

        hd1k = HD1K(aug_params=aug_params)

        train_dataset = 20 * sintel_clean + 20 * sintel_final + 200 * kitti + 5 * hd1k + things_clean
        print(len(sintel_clean), len(sintel_final), len(kitti), len(hd1k), len(things_clean))

    elif stage == 'viper':
        crop_size = (368, 768)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        aug_params = {'crop_size': crop_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        kitti = KITTI(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}

        hd1k = HD1K(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        viper = VIPER(aug_params=aug_params)

        # train_dataset = 80 * sintel_clean + 80 * sintel_final + 400 * kitti + 20 * hd1k + 4 * things_clean + viper
        train_dataset = 40 * sintel_clean + 40 * sintel_final + 20 * hd1k + 4 * things_clean + viper + 400 * kitti
        print(len(sintel_clean), len(sintel_final), len(kitti), len(hd1k), len(things_clean), len(viper))

    elif stage == 'kitti':
        aug_params = {'crop_size': (320, 1152), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}

        train_dataset = KITTI(aug_params, split='training')

    elif stage == 'neusim':
        crop_size = (320, 896)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass')

        aug_params = {'crop_size': crop_size, 'min_scale': -1, 'max_scale': 0, 'do_flip': False}
        neu_dataset = NeuSim(aug_params)

        train_dataset = things_clean + things_final + 2 * neu_dataset

    return train_dataset