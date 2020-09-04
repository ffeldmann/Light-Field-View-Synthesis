from torch.utils.data import Dataset
import os
import numpy as np
from scipy import misc

class LFData(Dataset):
    def __init__(self, path, split='train'):
        '''
        Loads the data from given folder in self.data
        The process is slow, needs improvements.
        I hope after loadind that will work fine in training
        '''
        self.path = path
        self.data_path = os.path.join(path, split)
        self.hor_path = os.path.join(self.data_path, 'h')
        self.vert_path = os.path.join(self.data_path, 'v')
        folders = os.listdir(self.hor_path)
        self.len = len(folders)
        data = {'h': [], 'v': []}
        hor_vol = np.zeros((9, 48, 48, 3))
        vert_vol = np.zeros((9,48,48,3))
        #import pdb;pdb.set_trace()
        for folder in folders:
            for idx in range(9):
                hor_vol[idx] = misc.imread(os.path.join(self.hor_path, folder, '{}.png'.format(idx)))
                vert_vol[idx] = misc.imread(os.path.join(self.vert_path, folder, '{}.png'.format(idx)))
            data['h'].append(hor_vol)
            data['v'].append(vert_vol)

        self.data = data

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        '''
        Returns a dictionary of horizontal and vertical
        crossing epipolar volumes, each of shape 9x48x48x3,
        where horizontal[4]==vertical[4]
        '''
        return {
            'horizontal': self.data['h'][i],
            'vertical': self.data['v'][i]
        }

