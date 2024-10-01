import numpy as np


def crop_for_ddd(img, patch_size=111, stride=33):
    m = (patch_size-1)//2
    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    elif len(img.shape) == 2:
        rows, cols = img.shape
    # rows, cols = img.shape

    mids = (stride-1)//2

    rowmax = np.arange(m+1, rows-m + 1, stride)
    colmax = np.arange(m+1, cols-m + 1, stride)

    img_crop = img[rowmax[0]-mids:rowmax[-1] +
                   mids+1, colmax[0]-mids:colmax[-1]+mids+1]
    return img_crop
