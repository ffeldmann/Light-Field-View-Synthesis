import os
import pickle

import cv2
import numpy as np
import albumentations

from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.util import adjust_support


class DeepfashionDataset(MetaDataset):
    def __init__(self, split="train"):
        self.data_dir = "3DCV_Datasets/deepfashion_vunet/"
        self.index_p = "3DCV_Datasets/deepfashion_vunet/index.p"

        # filter out valid indices based on valid_joints heuristic
        data = pickle.load(open(self.index_p, "rb"))

        # boolean list indicating if
        # element is from train or validations set
        self.is_train = data["train"].copy()

        # list of strings for filenames
        self.imgs = data["imgs"].copy()

        # filter out elements based on valid
        valid_train = np.array(self.is_train)
        valid_test = np.logical_not(np.array(self.is_train))

        if split == "train":
            self.imgs = list(np.array(self.imgs)[valid_train])
        else:
            self.imgs = list(np.array(self.imgs)[valid_test])
        self.rescale = albumentations.augmentations.transforms.Resize(128, 128)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.imgs[idx])
        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.rescale(image=image)["image"]
        image = adjust_support(image, "-1->1", "0->255")

        # build example for output
        example = {
            "image_path": image_path,
            "inp": image,
            "targets": image,
        }
        return example


class DeepfashionTrain(DatasetMixin):
    def __init__(self, config):
        super(DeepfashionTrain, self).__init__()
        self.config = config
        self.dset = DeepfashionDataset(split="train")

    def __len__(self):
        return len(self.dset)

    def get_example(self, idx):
        return self.dset[idx]


class DeepfashionVal(DeepfashionTrain):
    def __init__(self, config):
        self.config = config
        self.dset = DeepfashionDataset(split="valid")
