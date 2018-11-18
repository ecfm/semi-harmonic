from __future__ import division
from scipy.io import loadmat
import numpy as np
import os


def load_mnist(data_dir, dtype="float64"):
    """
    Load matrices from data_dir. Return matrices with dtype
    """
    train_x = np.load(os.path.join(data_dir, "train_x.npy")).astype(dtype)

    train_y = np.load(os.path.join(data_dir, "train_y.npy")).astype("int64")

    # Shuffle the data and normalize
    p_ix = np.random.permutation(train_x.shape[0])
    train_x = train_x[p_ix] / 255.
    train_y = train_y[p_ix]

    # Now return
    return train_x, train_y.flatten()