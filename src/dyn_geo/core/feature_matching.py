import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d
from bokeh.layouts import row

from dyn_geo.core import img
from dyn_geo.core.mask import masks_from_rois



def set_matcher(type_matching):
    if type_matching == 'flann':
        # Match with FLANN (faster than BFMatcher at scale)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif type_matching == 'bfmatcher':
        matcher = cv2.BFMatcher()
    return matcher


def get_matching_pts(matcher, des_ref, des):
    try:
        raw_matches = matcher.knnMatch(des_ref, des, k=2)
    except:
        raw_matches = []
    return raw_matches


def save_matches(src_pts, dst_pts, dir_matches, stem, valid_mask=None):
    dst_pts = np.squeeze(dst_pts)
    src_pts = np.squeeze(src_pts)

    data = {}
    data['src_x'] = src_pts[:, 0]
    data['src_y'] = src_pts[:, 1]
    data['dst_x'] = dst_pts[:, 0]
    data['dst_y'] = dst_pts[:, 1]
    if valid_mask is not None:
        data['valid'] = valid_mask.astype(bool)

    df = pd.DataFrame(data)

    # Save to CSV
    name = stem + '.csv'
    df.to_csv(dir_matches / name, index=False)
    return


def read_matches(dir_matches):

    matches = []
    ls_csv = sorted(dir_matches.glob('*.csv'))
    for f in ls_csv:
        matches.append(pd.read_csv(f))

    return ls_csv, matches


def plot_src_and_dst_matches_mpl(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, stem):
    dst_pts = np.squeeze(dst_pts)
    src_pts = np.squeeze(src_pts)
    inlier_mask = np.squeeze(inlier_mask)
    # inlier_inds = np.where(inlier_mask == 1)
    inlier_inds = np.where(inlier_mask)
    fig, ax = plt.subplots(1, 2, figsize=(22, 12), sharex=True, sharey=True, tight_layout=True)
    ax[0].set_title('Reference image')
    ax[0].imshow(im_ref)
    ax[1].set_title('Target image')
    ax[1].imshow(im)
    ax[0].plot(dst_pts[:, 0], dst_pts[:, 1], c='r', linewidth=0, markersize=6, marker='s')
    ax[0].plot(dst_pts[inlier_inds, 0], dst_pts[inlier_inds, 1], c='b', linewidth=0, markersize=6, marker='s')
    labels = np.arange(len(dst_pts[inlier_inds]))
    [ax[0].text(xi, yi, label, fontsize=10, ha='center', va='bottom') for xi, yi, label in zip(
        np.squeeze(dst_pts[inlier_inds, 0]), np.squeeze(dst_pts[inlier_inds, 1]), labels)]
    ax[1].plot(src_pts[:, 0], src_pts[:, 1], c='r', linewidth=0, markersize=6, marker='d', label='matches')
    ax[1].plot(src_pts[inlier_inds, 0].reshape(-1), src_pts[inlier_inds, 1].reshape(-1), c='b', linewidth=0,
               markersize=6, marker='d', label='valid matches (ransac)')
    ax[1].legend(loc='upper right')
    [ax[1].text(xi, yi, label, fontsize=10, ha='center', va='bottom') for xi, yi, label in zip(
        np.squeeze(src_pts[inlier_inds, 0]), np.squeeze(src_pts[inlier_inds, 1]), labels)]
    # rm x,y ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.savefig(outdir_matches_plots / (stem + '.jpg'), bbox_inches='tight')
    plt.close('all')


def plot_src_and_dst_matches(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, name, h, w):

    # convert images to rgba
    im_ref = img.to_rgba(im_ref, h, w)
    im = img.to_rgba(im, h, w)

    dst_pts = np.squeeze(dst_pts)
    src_pts = np.squeeze(src_pts)
    inlier_mask = np.squeeze(inlier_mask)
    inlier_inds = np.where(inlier_mask)

    # Flip y-coordinates to match Bokeh's bottom-left origin
    dst_pts[:, 1] = h - dst_pts[:, 1]
    src_pts[:, 1] = h - src_pts[:, 1]

    # Range1d objects to share the same ranges between p1 and p2
    x_range = Range1d(0, w)
    y_range = Range1d(0, h)

    # figure ref img keypoints
    p1 = figure(width=900, height=550, title="Reference image (dst_pts)", x_range=x_range, y_range=y_range)
    p1.image_rgba(image=[im_ref], x=0, y=0, dw=w, dh=h)

    # figure moving img keypoints
    p2 = figure(width=900, height=550, title="Current image (src_pts)", x_range=x_range, y_range=y_range)
    p2.image_rgba(image=[im], x=0, y=0, dw=w, dh=h)

    # Outliers in red
    outlier_inds = np.where(inlier_mask == 0)
    p2.scatter(src_pts[outlier_inds, 0].ravel(), src_pts[outlier_inds, 1].ravel(),color="red", size=6, marker="diamond")
    p1.scatter(dst_pts[outlier_inds, 0].ravel(), dst_pts[outlier_inds, 1].ravel(), color="red", size=6, marker="square")

    # Inliers in blue
    p2.scatter(src_pts[inlier_inds, 0].ravel(), src_pts[inlier_inds, 1].ravel(), color="blue", size=6, marker="diamond")
    p1.scatter(dst_pts[inlier_inds, 0].ravel(), dst_pts[inlier_inds, 1].ravel(), color="blue", size=6, marker="square")

    # plot inliers ids
    labels = np.arange(len(dst_pts[inlier_inds]))
    p2.text(
        x=src_pts[inlier_inds, 0].ravel(), y=src_pts[inlier_inds, 1].ravel(), text=labels, text_font_size='10pt',
        text_align='center', text_baseline='bottom')
    p1.text(
        x=dst_pts[inlier_inds, 0].ravel(), y=dst_pts[inlier_inds, 1].ravel(), text=labels, text_font_size='10pt',
        text_align='center', text_baseline='bottom')

    # Hide ticks, labels, axis line
    p1.xaxis.visible = False
    p1.yaxis.visible = False
    p2.xaxis.visible = False
    p2.yaxis.visible = False

    layout = row(p1, p2)
    name = name + '.html'
    output_file(outdir_matches_plots / name)
    save(layout)


def run(ref_fn, ref_f_rois, target_imgs_dir, start, end, f_cam_params, type_matching, path):

    # read reference image
    _, im_ref_gray, h, w = img.read_im(ref_fn, f_cam_params)

    # get masks from rois that were defined on ref image
    masks, _ = masks_from_rois(ref_f_rois, (h, w))

    # Create a SIFT object (is an algorithm used to detect and describe local features in images.
    # SIFT is robust to changes in scale, rotation, and illumination)
    sift = cv2.SIFT_create()

    # matcher
    matcher = set_matcher(type_matching) # type_matching = 'flann' or 'bf'

    # compute keypoints and descriptors on ref img
    kps_ref = []
    des_ref = []
    for i in range(len(masks)):
        kp, des = sift.detectAndCompute(im_ref_gray, masks[i])
        kps_ref.append(kp)
        des_ref.append(des)

    # dilate masks for target images
    sz_dil = 800
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz_dil, sz_dil))
    masks_target = [cv2.dilate(masks[i], kernel, iterations=1) for i in range(len(masks))]

    # loop through target images
    ls = img.ls_period(target_imgs_dir, start, end)

    for f in ls:
        print(f)

        # initialize source and destination points
        dst_pts = []
        src_pts = []

        # read target image
        _, im_gray, _, _ = img.read_im(f, f_cam_params)

        # loop through masks applied on target img
        for i in range(len(masks)):
            # compute keypoints and descriptors on target img
            kp, des = sift.detectAndCompute(im_gray, masks_target[i])
            # get matching keypoints
            raw_matches = get_matching_pts(matcher, des_ref[i], des)
            if len(raw_matches) > 0:
                # filter matching keypoints by a distance criteria
                good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
                # append src and dst pts
                dst_pts.append(np.float32([kps_ref[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2))
                src_pts.append(np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))

        # combine multiple arrays of shape (n, 1, 2) into a single array of shape (total_n, 1, 2)
        dst_pts = np.vstack(dst_pts)
        src_pts = np.vstack(src_pts)

        # save matches
        save_matches(src_pts, dst_pts, path.matches_data_raw, f.stem)

    return


def plot(fp_ref_im, target_imgs_dir, start, end, f_cam_params, dir_matches_data, dir_matches_plot):

    # list of csv matching points data files
    ls = img.ls_period(dir_matches_data, start, end, extension='*.csv')

    # read ref im
    im_ref, _, h, w = img.read_im(fp_ref_im, f_cam_params)

    for f_match in ls:

        # read csv
        df = pd.read_csv(f_match)
        inlier_mask = df['valid']

        # extract src and dst points
        src_pts = df[['src_x', 'src_y']].to_numpy()
        dst_pts = df[['dst_x', 'dst_y']].to_numpy()

        # target im
        im, _, _, _ = img.read_im(target_imgs_dir / (f_match.stem + '.jpg'), f_cam_params)

        # plot matches (with matplotlib, and bokeh)
        plot_src_and_dst_matches_mpl(src_pts, dst_pts, inlier_mask, im_ref, im, dir_matches_plot, f_match.stem)
        plot_src_and_dst_matches(src_pts, dst_pts, inlier_mask, im_ref, im, dir_matches_plot, f_match.stem, h, w)

    return


