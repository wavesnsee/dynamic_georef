import cv2
import json

from dateutil import parser
import numpy as np
from georef.operators import Georef
import turbojpeg as tjpeg
jpeg = tjpeg.TurboJPEG()


def is_jpeg(buffer: bytes) -> bool:
    """Check if last two bytes of the file buffer are equal to b'\xff\xd9'"""
    return buffer[-2:] == b"\xff\xd9"


def read_jpeg(fname, pixel_format=tjpeg.TJPF_RGB) -> np.ndarray:
    fname = fname
    with fname.open("rb") as fp:
        buffer = fp.read()

    if not is_jpeg(buffer=buffer):
        raise ValueError(f"JPEG File '{fname}' is corrupted")
    im = jpeg.decode(buffer, pixel_format)

    return im


def read_im(f, f_cam_params):

    # read camera georef parameters
    georef_params = Georef.from_param_file(f_cam_params)

    # read img
    im = read_jpeg(f)

    # undistort img
    im =  cv2.undistort(im, georef_params.intrinsic.camera_matrix, georef_params.dist_coeffs)

    # convert to gray
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # width, height of image
    h, w = im.shape[0:2]

    return im, im_gray, h, w


def rescale(im, scale_percent):
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    im = cv2.resize(im, (width, height))
    return im, width, height


def to_rgba(img, h, w, bokeh_compatible=True):
    """Convert image array to RGBA uint32 for Bokeh image_rgba."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgba = np.ones((h, w), dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape((h, w, 4))
    view[:, :, :3] = img
    view[:, :, 3] = 255
    if bokeh_compatible:
        return np.flipud(rgba)
    else:
        return rgba


def get_date(fn):
    ymd = fn.stem.split('_')[-3]
    h = fn.stem.split('_')[-2]
    mn = fn.stem.split('_')[-1]
    date = f'{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]} {h}:{mn}'
    date = parser.parse(date)
    return date


def ls_period(dir, start, end, extension='*.jp*g'):
    ls = sorted(dir.glob(extension))
    t_im = [get_date(f) for f in ls]
    inds = [i for i, dt in enumerate(t_im) if start <= dt <= end]
    return np.array(ls)[inds].tolist()


def ls_im_2rgba(ls, georef_params, undistort_map, scale_percent, bokeh_compatible=True):

    # initialize output list of rgba images
    rgba = []

    # loop through images
    for i, f in enumerate(ls):

        # read image
        im = read_jpeg(f)

        # georef param
        gp = georef_params[i]
        gp._undistort_map_x = undistort_map[0]
        gp._undistort_map_y = undistort_map[1]

        # undistort image
        im = gp.undistort(im)

        # rescale image
        if scale_percent < 100:
            im, width, height = rescale(im, scale_percent)
        else:
            (height, width, _) = np.shape(im)

        # convert to rgba
        im_rgba = to_rgba(im, height, width, bokeh_compatible)
        rgba.append(im_rgba)


    return rgba, width, height
