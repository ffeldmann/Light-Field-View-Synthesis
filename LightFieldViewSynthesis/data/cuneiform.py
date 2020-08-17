import numpy as np
import sklearn.model_selection
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.util import adjust_support


class Cuneiform(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        self.crop = crop
        # works if dataroot like "VOC2011/cats_meta"
        self.animal = config["dataroot"].split("/")[1].split("_")[0]
        if "rescale_to" in self.config.keys():
            # TODO: Albumenations or scikit image rescaling!
            self.rescale = Rescale(self.config["rescale_to"])
        else:
            # Scaling to default size 128
            self.rescale = Rescale((128, 128))


class AnimalVOC2011_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.sc = Cuneiform(config)
        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        self.augmentation = config["augmentation"]
        self.aug_factor = 0.5
        if self.augmentation:
            # TODO: albumenations augmentations
            pass

        if mode != "all":
            dset_indices = np.arange(len(self.sc))
            train_indices, test_indices = sklearn.model_selection.train_test_split(dset_indices,
                                                                                   train_size=float(
                                                                                       config["train_size"]),
                                                                                   random_state=int(
                                                                                       config["random_state"]))
            if mode == "train":
                self.data = SubDataset(self.sc, train_indices)
            else:
                self.data = SubDataset(self.sc, test_indices)
        else:
            self.data = self.sc

    def get_example(self, idx: object) -> object:
        """
        Args:
            idx: integer indicating index of dataset

        Returns: example element from dataset

        """
        example = super().get_example(idx)
        image = example["frames"]()
        target = example["target"]()

        if self.augmentation:
            # randomly perform some augmentations on the image, keypoints and bboxes
            image = self.seq(image=image)

        # we always work with "-1->1" images and np.float32
        example["inp"] = image
        return example


class Cuneiform_Train(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")


class Cuneiform_Validation(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")
        self.augmentation = False


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys


    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            pdb.pm()


    sys.excepthook = info

    DATAROOT = {"dataroot": ''}
    cats = Cuneiform(DATAROOT)
    ex = cats.get_example(3)
    for hm in ex["targets"]:
        print(hm.shape)
        plt.imshow(hm)
        plt.show()
