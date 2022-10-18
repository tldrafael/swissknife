import torch
import numpy as np
import time
from glob import iglob


def keepNB_awake():
    while True:
        time.sleep(5)


def list_files(dirpath, fl_sort=True):
    fpaths = list(iglob(dirpath + '/*'))
    if fl_sort:
        fpaths.sort()
    return fpaths


def set_randomseed(seed=None, return_seed=False):
    if seed is None:
        seed = np.random.randint(2147483647)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if return_seed:
        return seed
