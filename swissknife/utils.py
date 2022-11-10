import seaborn as sns
import torch
import numpy as np
import time
from glob import iglob
import logging
from logging.handlers import RotatingFileHandler


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


class Logger():
    def __init__(self):
        self.root = logging.getLogger()
        self.root.setLevel(logging.DEBUG)
        self.dataset_header = []

        if self.root.hasHandlers():
            self.root.handlers.clear()

        self.handlers = [logging.StreamHandler(),
                         RotatingFileHandler('Trainer.log', mode='a', maxBytes=50e6, backupCount=2)]
        self.formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        for h in self.handlers:
            h.setFormatter(self.formatter)
            h.setLevel(logging.DEBUG)
            self.root.addHandler(h)

    def log(self, msg, level=logging.INFO):
        self.root.log(level, msg)

    def add_handler(self, logpath, fl_resume=False):
        wmode = 'a' if fl_resume else 'w'
        handler = RotatingFileHandler(logpath, mode=wmode, maxBytes=50e6, backupCount=2)
        handler.setFormatter(self.formatter)
        handler.setLevel(logging.DEBUG)
        self.root.addHandler(handler)

    def log_header(self):
        for msg in self.dataset_header:
            self.log(msg)


class LabelColorizer:
    def __init__(self, n_classes=12, ix_nolabel=255):
        assert isinstance(ix_nolabel, (int, type(None)))
        self.ix_nolabel = ix_nolabel
        self.n_classes = n_classes
        self.map = self.get_pallete()

    def get_pallete(self):
        pal = sns.color_palette(palette='gist_rainbow', as_cmap=True)(np.linspace(0, 1, self.n_classes))[..., :3]
        pal = np.vstack([[[0, 0, 0]], pal])
        dict_pal = {}
        for i in range(1, pal.shape[0]):
            dict_pal[i - 1] = torch.tensor(pal[i])

        if self.ix_nolabel is not None:
            dict_pal[self.ix_nolabel] = torch.tensor(pal[0])
        return dict_pal

    def __call__(self, mask):
        fl_single = False
        if len(mask.shape) < 4:
            fl_single = True
            mask = mask[None]

        bs = mask.shape[0]
        cmask = torch.zeros((bs, 3,) + mask.shape[2:]).type(self.map[1].dtype)
        for i in range(bs):
            for k in self.map.keys():
                cmask[i, :, mask[i, 0] == k] = self.map[k][:, None]

        if fl_single:
            cmask = cmask[0]
        return cmask

    def reverse(self, x):
        mask_new = torch.zeros(x.shape[1:])
        for k, v in self.map.items():
            k_pos = torch.all(torch.eq(x.float(), torch.Tensor(v)[:, None, None]), axis=0)
            mask_new[k_pos] = k
