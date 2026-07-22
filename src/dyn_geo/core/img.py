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


def read_jpeg(fname, pixel_format=tjpeg.TJPF_BGR) -> np.ndarray:
    fname = fname
    with fname.open("rb") as fp:
        buffer = fp.read()

    if not is_jpeg(buffer=buffer):
        raise ValueError(f"JPEG File '{fname}' is corrupted")
    bgr = jpeg.decode(buffer, pixel_format)

    # convert from bgr to rgb
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def read(f, f_cam_params):

    # read input img
    im = read_jpeg(f)

    # convert to gray
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # read camera georef parameters
    georef_params = Georef.from_param_file(f_cam_params)

    # undistort grayscale img
    im_gray = cv2.undistort(im_gray, georef_params.intrinsic.camera_matrix, georef_params.dist_coeffs)

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


def edges(gray):
    # https://www.geeksforgeeks.org/python/real-time-edge-detection-using-opencv-python/

    def apply_clahe(gray):
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def bilateral_smooth(gray):
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def dynamic_canny(smooth):
        sigma = np.std(smooth)
        lower = max(20, int(0.66 * sigma))
        upper = min(200, int(1.33 * sigma))
        return cv2.Canny(smooth, lower, upper, apertureSize=3, L2gradient=True)

    def sobel_gradient(gray):
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(np.sqrt(sx ** 2 + sy ** 2))

    def laplacian_edge(gray):
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    def fuse_edges(canny, lap, sobel):
        fused = cv2.addWeighted(canny, 0.6, lap, 0.3, 0)
        return cv2.addWeighted(fused, 0.7, sobel, 0.3, 0)

    def morphology_close(fused):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)

    def overlay_edges(frame, fused):
        overlay = frame.copy()
        overlay[fused > 80] = [0, 0, 255]
        return overlay

    def process_frame(gray):
        clahe_gray = apply_clahe(gray)
        smooth = bilateral_smooth(clahe_gray)
        canny = dynamic_canny(smooth)
        lap = laplacian_edge(smooth)
        sobel = sobel_gradient(smooth)
        fused = fuse_edges(canny, lap, sobel)
        fused = morphology_close(fused)
        return fused

    return process_frame(gray)
