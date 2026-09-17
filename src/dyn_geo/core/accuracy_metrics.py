import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d, RangeTool
from bokeh.layouts import column
from bokeh.models import Div
from matplotlib.path import Path
import numpy as np
import cv2

from dyn_geo.core import img


def get_n_matching_pts(ls):
    n = []
    n_valid = []
    t = []
    for f_match in ls:
        df = pd.read_csv(f_match)
        date = img.get_date(f_match)
        t.append(date)
        n.append(len(df))
        n_valid.append(np.sum(df['valid']))
    return t, n, n_valid


def reprojection_error(
        ls_match_pts: list,
        dir_h: list
) -> dict:
    """
    Compute reprojection error between source and destination points
    given a homography matrix H.

    Args:
        src_pts: Source points, shape (N, 2) or (N, 1, 2)
        dst_pts: Destination points, shape (N, 2) or (N, 1, 2)
        H:       3x3 homography matrix (maps src → dst)

    Returns:
        dict of stats
    """
    mean = []
    std = []
    max = []
    rmse = []
    t = []

    for f_match in ls_match_pts:

        # read csv
        df = pd.read_csv(f_match)

        # keep only valid (from ransac) matching points
        df = df[df['valid']]

        # extract src and dst points
        src = df[['src_x', 'src_y']].to_numpy()
        dst = df[['dst_x', 'dst_y']].to_numpy()

        # read h
        f_h = dir_h / (f_match.stem + '.npy')

        if f_h.exists:

            # load homography matrix
            H = np.load(f_h)
            assert H.shape == (3, 3), "H must be a 3x3 matrix"

            # Project src → dst space via H
            projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H)
            projected = projected.reshape(-1, 2)

            # date
            date = img.get_date(f_match)

            # Per-point Euclidean distance
            errors = np.linalg.norm(projected - dst, axis=1)

            # store errors' statistics
            t.append(date)
            mean.append(float(np.mean(errors)))
            std.append(float(np.std(errors)))
            max.append(float(np.max(errors)))
            rmse.append(np.sqrt(np.mean(errors ** 2)))

    return {
          "t" : t,
        "mean": mean,
         "std": std,
         "max": max,
        "rmse": rmse,
    }

def compute_err_metrics(
        ref_fname: Path,
        target_imgs_dir: Path,
        dir_h: Path,
        f_cam_params: Path
):

    # initialization
    t = []
    phase_res = []

    # read ref im
    im_ref, im_ref_gray, h, w = img.read_im(ref_fname, f_cam_params)

    # list of homography matrixes' files
    ls = sorted(dir_h.glob('*.npy'))

    for f_h in ls:

        # load homography matrix
        H = np.load(f_h)

        # read target im
        im, im_gray, _, _ = img.read_im(target_imgs_dir / (f_h.stem + '.jpg'), f_cam_params)

        # warp im
        warped_im = cv2.warpPerspective(im, H, (w, h))
        warped_im_gray = cv2.cvtColor(warped_im, cv2.COLOR_BGR2GRAY)

        # date
        date = img.get_date(f_h)
        t.append(date)

        # phase residual
        shift_px, response = phase_residual(im_ref_gray, warped_im_gray)
        phase_res.append(shift_px)

        # Structural Similarity Index
        # tester des methodes plus robustes que skimage.metrics import structural_similarity as ssim
        # (voir partie "Deep Learning-Based Approaches"):
        # https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3


    return t, phase_res


def phase_residual(ref_gray, warped_gray):
    (dx, dy), response = cv2.phaseCorrelate(
        ref_gray.astype(np.float64),
        warped_gray.astype(np.float64)
    )
    shift_px = np.sqrt(dx**2 + dy**2)
    return shift_px, response  # bad if shift_px > 1–2px or response < 0.05


def run(dir_matches_data, dir_h, dir_acc_metrics, ref_fname, target_imgs_dir, f_cam_params):

    # list of csv matching points data files
    ls = sorted(dir_matches_data.glob('*.csv'))

    # get number of matching points
    t, n, n_valid = get_n_matching_pts(ls)

    # compute reprojection errors
    errors = reprojection_error(ls, dir_h)

    # Error metrics between ref image and target images
    t_error_metrics, phase_res = compute_err_metrics(ref_fname, target_imgs_dir, dir_h, f_cam_params)

    # Create a global title using a Div
    global_title = Div(text="<h1>Homography accuracy metrics</h1>", sizing_mode='stretch_width')

    # Range1d objects to share the same ranges between p1 and p2
    x_range = Range1d(min(t), max(t))

    plot_h = 200
    p1 = figure(sizing_mode='stretch_width', height=plot_h, title="Number of matching points", x_range=x_range)
    p1.line(t, n, legend_label="raw", line_color="red", line_width=2)
    p1.line(t, n_valid, legend_label="valid", line_color="blue", line_width=2)
    p1.yaxis.axis_label = 'N matching points'

    p2 = figure(sizing_mode='stretch_width', height=plot_h, title="Reprojection error", x_range=x_range)
    p2.line(errors['t'], errors['max'], legend_label="max", line_color="chocolate", line_width=2)
    p2.line(errors['t'], errors['mean'], legend_label="mean", line_color="blue", line_width=2)
    p2.line(errors['t'], errors['std'], legend_label="std", line_color="black", line_width=2)
    p2.line(errors['t'], errors['rmse'], legend_label="rmse", line_color="mediumslateblue", line_width=2)
    p2.yaxis.axis_label = 'Reprojection errors (pixels)'

    p3 = figure(sizing_mode='stretch_width', height=plot_h, title="Phase residual",
                x_range=x_range)
    p3.line(t_error_metrics, phase_res, legend_label="Phase residual", line_color="black", line_width=2)

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    sizing_mode='stretch_width', height=plot_h,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p2.x_range, start_gesture="pan")
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(t, errors['max'])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)


    name = 'homography_accuracy_metrics.html'
    output_file(dir_acc_metrics / name, title='ACCURACY METRICS')
    layout = column(global_title, p1, p2, p3, select, sizing_mode='stretch_width')
    save(layout)

    return