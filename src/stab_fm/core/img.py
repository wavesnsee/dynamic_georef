import cv2
import json
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt


def read(f, f_calib):

    # read input img
    im = cv2.imread(f)

    # convert from bgr to rgb
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # convert to gray
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # read camera matrix and distorsion coeffs
    K, dist_coeffs = read_camera_matrix_and_dist(f_calib)

    # undistort grayscale img
    im_gray = cv2.undistort(im_gray, K, dist_coeffs)

    # width, height of image
    h, w = im.shape[0:2]

    return im, im_gray, h, w


def to_rgba(img, h, w):
    """Convert image array to RGBA uint32 for Bokeh image_rgba."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgba = np.ones((h, w), dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape((h, w, 4))
    view[:, :, :3] = img
    view[:, :, 3] = 255
    return rgba


def read_camera_matrix_and_dist(f_calib):
    K = read_json(f_calib)["camera_matrix"]
    dist_coeffs = read_json(f_calib)["dist_coeffs"]
    return K, dist_coeffs


def read_json(fn):
    """

    Parameters
    ----------
    fn : string
        name of the file to load

    Returns
    -------
    dict : dictionnary containing numpy arrays
    """
    try:
        with open(fn, 'r') as infile:
            dict = json.load(infile)
            for k in dict.keys():
                if type(dict[k]) is list:
                    dict[k] = np.array(dict[k])
            return dict
    except IOError:
        raise


def get_date(fn):
    ymd = fn.stem.split('_')[-3]
    h = fn.stem.split('_')[-2]
    mn = fn.stem.split('_')[-1]
    date = f'{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]} {h}:{mn}'
    date = parser.parse(date)
    return date
