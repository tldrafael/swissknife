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
