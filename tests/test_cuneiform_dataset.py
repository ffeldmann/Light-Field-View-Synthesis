from LightFieldViewSynthesis.data.cuneiform import Cuneiform_Train, Cuneiform_Validation
from tqdm import tqdm
config = {
    "n_classes": 19,
    "n_channels": 3,  # rgb or grayscale
    "bilinear": True,
    "resize_to": 128,
    "as_grey": False,
    "sigma": 3,  # Good way sigma = size/64 for heatmaps
    "crop": True,
    "augmentation": False,
    "train_size": 0.8,
    "random_state": 42,
    "pck_alpha": 0.5,
    "pck_multi": False,
    "sequence_step_size": 1, # Steps to be taken from one frame to another
    "cropped": True # loads the bigger {animal}_cropped dataset
}

dtrain = Cuneiform_Train(config)
dtest = Cuneiform_Validation(config)


def test_get_example():
    # for element in dataset: print(element["frames"]().shape)
    ex = dtrain.get_example(5)

def test_fid_different_dtrain():
    print()
    for element in tqdm(dtrain):
        assert element["fid0"] != element["fid1"]

def test_fid_different_dtest():
    print()
    for element in tqdm(dtest):
        assert element["fid0"] != element["fid1"]

def test_all_shapes():
    size = config["resize_to"]
    print()
    for element in tqdm(dtrain):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
        width, height, _ = element["inp1"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
    print()
    for element in tqdm(dtest):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
        width, height, _ = element["inp1"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
