from torch.utils.data import Dataset
import os
import numpy as np
from scipy import misc
import imageio
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.util import adjust_support
from tqdm import tqdm


class LFData(MetaDataset):
    def __init__(self, split='train'):
        '''
        Loads the data from given folder in self.data
        The process is slow, needs improvements.
        I hope after loadind that will work fine in training
        '''
        self.path = '/export/home/tmuradya/3dcv/data_folder/'
        self.data_path = os.path.join(self.path, split)
        self.hor_path = os.path.join(self.data_path, 'h')
        self.vert_path = os.path.join(self.data_path, 'v')
        folders = os.listdir(self.hor_path)
        data = {'h': [], 'v': []}
        for folder in tqdm(folders):
            hor_vol = []  # np.zeros((9, 48, 48, 3))
            vert_vol = []  # np.zeros((9,48,48,3))
            for idx in range(9):
                hor_vol.append(os.path.join(self.hor_path, folder, '{}.png'.format(idx)))
                vert_vol.append(os.path.join(self.vert_path, folder, '{}.png'.format(idx)))
            data['h'].append(hor_vol)
            data['v'].append(vert_vol)
        self.len = len(data['h'])

        self.data = data

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        '''
        Returns a dictionary of horizontal and vertical
        crossing epipolar volumes, each of shape 9x48x48x3,
        where horizontal[4]==vertical[4]
        '''

        hor_vol = np.zeros((9, 48, 48, 3))
        vert_vol = np.zeros((9,48,48,3))
        hor_coords = np.zeros((9))
        vert_coords = np.zeros((9))

        for idx, img in enumerate(self.data['h'][i]):
            hor_vol[idx] = adjust_support(imageio.imread(img), "-1->1", "0->255")
            hor_coords[idx] = (0.5, float(idx)/9)
        for idx, img in enumerate(self.data['v'][i]):
            vert_vol[idx] = adjust_support(imageio.imread(img), "-1->1", "0->255")
            vert_coords[idx] = (float(idx)/9, 0.5)


        return {
            'horizontal': hor_vol,
            'horizontal_coords': hor_coords,
            'vertical': vert_vol,
            'vertical_coords': vert_coords,

        }


class LFDataTrain(DatasetMixin):
    def __init__(self, config):
        super(LFDataTrain, self).__init__()
        self.config = config
        self.dset = LFData(split="train")

    def __len__(self):
        return len(self.dset)

    def get_example(self, idx):
        return self.dset[idx]


class LFDataTest(LFDataTrain):
    def __init__(self, config):
        self.config = config
        self.dset = LFData(split="test")
