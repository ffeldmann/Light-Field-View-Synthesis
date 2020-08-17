import numpy as np
import torch


def normalize_nparray(array: np.array):
    """
        Normalize array to 0->1
        Args:
            array:

        Returns:

        """
    array -= array.min(1, keepdims=True)[0]
    array /= array.max(1, keepdims=True)[0].clip(min=1e-6)
    return array


def normalize_tensor(tensor: torch.tensor):
    """
    Normalize tensor to 0->1
    Args:
        tensor:

    Returns:

    """

    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]
    return tensor
