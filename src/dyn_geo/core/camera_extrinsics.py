import json
from copy import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from georef.operators import Georef, ExtrinsicMatrix

from dyn_geo.core import img


def plot_gcps_ref_target(gcps_uv, gcps_uv_warped, f_cam_params, target_img_fn, ref_img_fn, dir_gcps):

    # read reference image
    im_ref, _, _, _ = img.read(ref_img_fn, f_cam_params)

    # read target img
    im, _, _, _ = img.read(target_img_fn, f_cam_params)

    # plot gcps on reference image and on target image
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))
    ax[0].imshow(im_ref)
    ax[0].set_title('Reference Image')
    ax[0].plot(gcps_uv[:, 0], gcps_uv[:, 1], c='r', linewidth=0, markersize=3, marker='s', label='gcps raw')
    ax[0].legend(loc='upper right')
    ax[1].imshow(im)
    ax[1].plot(np.squeeze(gcps_uv_warped)[:, 0], np.squeeze(gcps_uv_warped)[:, 1], c='r', linewidth=0, markersize=3,
               marker='s', label='gcps warped with H')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Target Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.savefig(dir_gcps / target_img_fn.name, bbox_inches='tight')

    return


def compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, target_imgs_dir, ref_img_fn, dir_gcps, start, end,
                              outdir_cam_params_upd, plot_gcps=True):

    # list of homography matrixes
    # ls_h = sorted(dir_h.glob('*.npy'))
    ls_h = img.ls_period(dir_h, start, end, extension='*.npy')

    # Read initial camera parameters file
    with open(f_cam_params, 'r') as f:
        cam_params = json.load(f)

    # read georef parameters, that will be updated for each target image
    georef_params = Georef.from_param_file(f_cam_params)

    # initialize list of Georef objects
    georef_params_upd = [copy(georef_params) for i in range(len(ls_h))]

    # initialize date
    # date = []

    # read gcps file
    df = pd.read_csv(f_gcps)

    # extract gcps pixel coordinates
    gcps_uv = df[['U', 'V']].to_numpy()

    # compute gcps geo coordinates in local srs
    gcps_xyz = df[['easting', 'northing', 'elevation']].to_numpy().T
    gcps_xyz = (georef_params.local_srs.m_l_w @ gcps_xyz).T[:, 0:3]

    # reshape gcps_xyz to make it compatible with solvePnPRansac
    gcps_xyz = gcps_xyz.reshape(gcps_xyz.shape[0], 1, gcps_xyz.shape[1])

    # loop through homographies
    for i, f_h in enumerate(ls_h):
        # load homography matrix
        H = np.load(f_h)

        # reverse H
        H = np.linalg.inv(H)

        # apply homography to gcps
        gcps_uv_warped = cv2.perspectiveTransform(gcps_uv.reshape(-1, 1, 2), H)
        gcps_uv_warped = gcps_uv_warped.reshape(-1, 2)
        gcps_uv_warped = gcps_uv_warped.reshape(gcps_uv_warped.shape[0], 1, gcps_uv_warped.shape[1])

        if plot_gcps:
            # plot gcps on reference image and on target image
            target_img_fn = target_imgs_dir / f_h.name.replace('.npy', '.jpg')
            plot_gcps_ref_target(gcps_uv, gcps_uv_warped, f_cam_params, target_img_fn, ref_img_fn, dir_gcps)


        # compute dynamic georef from warped gcps
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(gcps_xyz.astype(np.float32),
                                                      gcps_uv_warped.astype(np.float32),
                                                      georef_params.intrinsic_parameters.camera_matrix,
                                                      georef_params.distortion_coefficients.array,
                                                      rvec=None,
                                                      tvec=None,
                                                      iterationsCount=50000,
                                                      reprojectionError=2,
                                                      flags=cv2.SOLVEPNP_EPNP)

        # time
        # t = img.get_date(f_h)
        # date.append(t)

        # save updated georef parameters
        extrinsic_upd = ExtrinsicMatrix(rvec, tvec)
        georef_params_upd[i].extrinsic = extrinsic_upd

        # save updated camera parameters, changing only extrinsic parameters
        cam_params['extrinsic_parameters']['rvec'] = rvec.reshape(-1).tolist()
        cam_params['extrinsic_parameters']['tvec'] = tvec.reshape(-1).tolist()
        with open(outdir_cam_params_upd / f_h.name.replace('.npy', '.json'), 'w') as f:
            json.dump(cam_params, f, indent=2)
    # return date, georef_params_upd
    return


def read_cam_params(dir_cparams, start=None, end=None):

    # initialize georef_params and date
    georef_params = []
    t_cparams = []

    # list of json camera parameters
    ls_cparams = sorted(dir_cparams.glob('*.json'))

    # read camera parameters
    for f in ls_cparams:
        date = img.get_date(f)
        if (start is None) and (end is None):
            gp = Georef.from_param_file(f)
            georef_params.append(gp)
            t_cparams.append(date)
        else:
            if (date >= start) and (date <= end):
                gp = Georef.from_param_file(f)
                georef_params.append(gp)
                t_cparams.append(date)

    return t_cparams, georef_params


def run(dir_h, dir_imgs, ref_img_fn, f_gcps, f_cam_params, start, end, dir_gcps, odir_cparams):

    # compute georef parameters for each target image
    compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, dir_imgs, ref_img_fn, dir_gcps,
                                                    start, end, odir_cparams)



