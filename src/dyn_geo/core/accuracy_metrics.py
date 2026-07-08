import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d, RangeTool
from bokeh.layouts import column
from matplotlib.path import Path
from skimage.metrics import structural_similarity as ssim

from dyn_geo.core import img
from dyn_geo.core.mask import masks_from_rois

import numpy as np
import cv2


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

def compute_error_metrics(
        ref_fname: Path,
        target_imgs_dir: Path,
        dir_h: Path,
        ref_f_rois: Path
):

    # initialize psnr and ssi lists
    psnr = []
    t = []
    ssi = []
    phase_res = []
    edge_score = []

    # read ref im
    im_ref = cv2.imread(ref_fname)
    im_ref_gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)
    h, w = im_ref.shape[0:2]

    # get masks from rois that were defined on ref image
    masks, mask_ref = masks_from_rois(ref_f_rois, (h, w))
    mask_ref = (mask_ref / 255).astype(np.uint8)

    # list of homography matrixes' files
    ls = sorted(dir_h.glob('*.npy'))

    for f_h in ls:

        # load homography matrix
        H = np.load(f_h)

        # read target im
        im = cv2.imread(target_imgs_dir / (f_h.stem + '.jpg'))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # warp im
        warped_im = cv2.warpPerspective(im, H, (w, h))
        warped_im_gray = cv2.cvtColor(warped_im, cv2.COLOR_BGR2GRAY)

        # date
        date = img.get_date(f_h)
        t.append(date)

        # phase residual
        shift_px, response = phase_residual(im_ref_gray, warped_im_gray)
        phase_res.append(shift_px)

        # edge score
        score, e1, e2 = edge_alignment_score(im_ref_gray, warped_im_gray, mask=mask_ref)
        edge_score.append(score)

        # Peak Signal-to-Noise Ratio
        psnr.append(cv2.PSNR(im_ref, warped_im))

        # Structural Similarity Index
        # tester des methodes plus robustes (voir partie "Deep Learning-Based Approaches"):
        # https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3

        try:
            ssim_score, diff = ssim(im_ref_gray, warped_im_gray, full=True, data_range=255)
            ssim_score = np.mean(diff[mask_ref==1])
            ssi.append(ssim_score)
        except:
            ssi.append(None)

    return t, psnr, ssi, phase_res, edge_score


def phase_residual(ref_gray, warped_gray):
    (dx, dy), response = cv2.phaseCorrelate(
        ref_gray.astype(np.float64),
        warped_gray.astype(np.float64)
    )
    shift_px = np.sqrt(dx**2 + dy**2)
    return shift_px, response  # bad if shift_px > 1–2px or response < 0.05


def edge_alignment_score(img1, img2, mask=None):

    e1 = img.edges(img1)
    e2 = img.edges(img2)

    if mask is not None:
        e1, e2 = e1 * mask, e2 * mask

    # Dilate to allow sub-pixel tolerance
    e1d = cv2.dilate(e1, np.ones((3,3)))
    e2d = cv2.dilate(e2, np.ones((3,3)))
    # plt.imshow(e1d)
    # plt.figure()
    # plt.imshow(e2d)
    # plt.show()
    recall    = (e1 * e2d).sum() / (e1.sum() + 1e-6)
    precision = (e2 * e1d).sum() / (e2.sum() + 1e-6)

    return 2 * recall * precision / (recall + precision + 1e-6), e1, e2  # F1



def run(dir_matches_data, dir_h, ref_f_rois_edges, dir_acc_metrics, ref_fname, target_imgs_dir):

    # list of csv matching points data files
    ls = sorted(dir_matches_data.glob('*.csv'))

    # get number of matching points
    t, n, n_valid = get_n_matching_pts(ls)

    # compute reprojection errors
    errors = reprojection_error(ls, dir_h)

    # Error metrics between ref image and target images
    t_error_metrics, psnr, ssi, phase_res, edge_score = compute_error_metrics(ref_fname,
                                                             target_imgs_dir,
                                                             dir_h,
                                                             ref_f_rois_edges)

    # Range1d objects to share the same ranges between p1 and p2
    x_range = Range1d(min(t), max(t))

    plot_h = 200
    plot_w = 1200
    p1 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="Number of matching points", x_range=x_range)
    p1.line(t, n, legend_label="raw", line_color="red", line_width=2)
    p1.line(t, n_valid, legend_label="valid", line_color="blue", line_width=2)
    p1.yaxis.axis_label = 'N matching points'

    p2 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="Reprojection error", x_range=x_range)
    p2.line(errors['t'], errors['max'], legend_label="max", line_color="chocolate", line_width=2)
    p2.line(errors['t'], errors['mean'], legend_label="mean", line_color="blue", line_width=2)
    p2.line(errors['t'], errors['std'], legend_label="std", line_color="black", line_width=2)
    p2.line(errors['t'], errors['rmse'], legend_label="rmse", line_color="mediumslateblue", line_width=2)
    p2.yaxis.axis_label = 'Reprojection errors (pixels)'

    p3 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="Phase residual",
                x_range=x_range)
    p3.line(t_error_metrics, phase_res, legend_label="Phase residual", line_color="black", line_width=2)

    p4 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="edge alignment score",
                x_range=x_range)
    p4.line(t_error_metrics, edge_score, legend_label="Edge alignment score", line_color="black", line_width=2)

    p5 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="Peak Signal-to-Noise Ratio",
                x_range=x_range)
    p5.line(t_error_metrics, psnr, legend_label="psnr", line_color="black", line_width=2)
    p5.yaxis.axis_label = 'psnr'

    p6 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", title="Structural_similarity index",
                x_range=x_range)
    p6.line(t_error_metrics, ssi, legend_label="ssi", line_color="black", line_width=2)
    p6.yaxis.axis_label = 'ssi'

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=plot_h, width=plot_w,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p2.x_range, start_gesture="pan")
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(t, errors['max'])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)


    name = 'accuracy_metrics.html'
    output_file(dir_acc_metrics / name)
    layout = column(p1, p2, p3, p4, p5, p6, select)
    save(layout)

    # plt.plot(t, n, 'r')
    # plt.plot(t, n_valid, 'b')
    # plt.show()


    return