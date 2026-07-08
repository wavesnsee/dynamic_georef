import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2
from copy import copy
import json
from georef.operators import Georef, ExtrinsicMatrix
from dyn_geo.core import img
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib.dates import date2num
from datetime import timedelta, datetime


def rotate_vector(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    return data.dot(rotation_matrix)


def plot_gcps_ref_target(gcps_uv, gcps_uv_warped, f_cam_params, target_img_fn, ref_img_fn, dir_gcps):

    # read reference image
    im_ref, _, _, _ = img.read(ref_img_fn, f_cam_params)

    # read target img
    im, _, _, _ = img.read(target_img_fn, f_cam_params)

    # plot gcps on reference image and on target image
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))
    ax[0].imshow(im_ref)
    ax[0].set_title('Reference Image')
    ax[0].plot(gcps_uv[:, 0], gcps_uv[:, 1], c='r', linewidth=0, markersize=3, marker='s', label='gcps')
    ax[0].legend(loc='upper right')
    ax[1].imshow(im)
    ax[1].plot(np.squeeze(gcps_uv_warped)[:, 0], np.squeeze(gcps_uv_warped)[:, 1], c='r', linewidth=0, markersize=3,
               marker='s', label='gcps')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Target Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.savefig(dir_gcps / target_img_fn.name, bbox_inches='tight')

    return



def compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, target_imgs_dir, ref_img_fn, dir_gcps, outdir_cam_params_upd,
                              plot_gcps=False):

    # list of homography matrixes
    ls_h = sorted(dir_h.glob('*.npy'))

    # Read initial camara_parameters file
    with open(f_cam_params, 'r') as f:
        cam_params = json.load(f)

    # read georef parameters, that will be updated for each target image
    georef_params = Georef.from_param_file(f_cam_params)

    # initialize list of Georef objects
    georef_params_upd = [copy(georef_params) for i in range(len(ls_h))]

    # initialize date
    date = []

    # read gcps file
    df = pd.read_csv(f_gcps)

    # extract gcps pixel coordinates
    gcps_uv = df[['U', 'V']].to_numpy()

    # extract gcps geo coordinates
    gcps_xy = df[['easting', 'northing']].to_numpy()
    gcps_xyz = df[['easting', 'northing', 'elevation']].to_numpy()

    # applying rotation to the local system
    data = np.stack((gcps_xy[:, 0] - georef_params.local_srs.offset_easting,
                     gcps_xy[:, 1] - georef_params.local_srs.offset_northing)).T
    rotated_data = rotate_vector(data, georef_params.local_srs.rotation)

    gcps_xyz[:, 0] = rotated_data[:, 0]
    gcps_xyz[:, 1] = rotated_data[:, 1]
    gcps_xyz = gcps_xyz.astype(np.float32)

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
        t = img.get_date(f_h)
        date.append(t)

        # save updated georef parameters
        extrinsic_upd = ExtrinsicMatrix(rvec, tvec)
        georef_params_upd[i].extrinsic = extrinsic_upd

        # save updated camera parameters, changing only extrinsic parameters
        cam_params['extrinsic_parameters']['rvec'] = rvec.reshape(-1).tolist()
        cam_params['extrinsic_parameters']['tvec'] = tvec.reshape(-1).tolist()
        with open(outdir_cam_params_upd / f_h.name.replace('.npy', '.json'), 'w') as f:
            json.dump(cam_params, f, indent=2)
    return date, georef_params_upd


# def interp_targets_extrinsic(odir_cparams_upd, odir_cparams_upd_smooth):
def interp_targets_extrinsic(dates, georef_params_upd, f_cam_params):
    '''
    Interpolation of rotation vector to compute smoothed camera parameters
    '''


    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation

    # read georef parameters, that will be updated for each target image
    georef_params = Georef.from_param_file(f_cam_params)

    # initialize variable rvecs
    rvecs = []

    def round_to_day_at_noon(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=0.5)

    for i in range(len(georef_params_upd)):
        rvecs.append(georef_params_upd[i].extrinsic.rvec.squeeze())

    # Initialize the multiple rotations in one object
    rots = R.from_rotvec(rvecs)

    # dates to perform interpolation
    interp_dates = np.arange(round_to_day_at_noon(dates[0]), round_to_day_at_noon(dates[-1]) + timedelta(days=1),
                             timedelta(days=1)).astype(datetime)
    interp_dates = [date2num(interp_dates[i]) for i in range(len(interp_dates))]

    # initialize list of Georef objects
    georef_params_interp = [copy(georef_params) for i in range(len(interp_dates))]

    # Create the interpolator object
    dates = [date2num(dates[i]) for i in range(len(dates))]
    slerp = Slerp(dates, rots)

    # perform Spherical Linear Interpolation of Rotations
    interp_rots = slerp(interp_dates)

    # save interp_rots as rotation vectors
    interp_rvecs = [interp_rots[i].as_rotvec() for i in range(len(interp_rots))]

    # save updated georef parameters
    for i in range(len(interp_rvecs)):
        extrinsic_interp = ExtrinsicMatrix(interp_rvecs[i], georef_params_upd[0].extrinsic.tvec)
        georef_params_interp[i].extrinsic = extrinsic_interp

    return interp_dates, georef_params_interp


def compute_cam_mvts(list_georef_params):

    angles = {}
    position = {}
    angles['yaw'] = []
    angles['pitch'] = []
    angles['roll'] = []
    position['x'] = []
    position['y'] = []
    position['z'] = []

    for georef_params in list_georef_params:

        # get camera angles and position
        a0, a1, a2 = georef_params.extrinsic.beachcam_angles
        px, py, pz = georef_params.extrinsic.camera_position
        angles['yaw'].append(a0)
        angles['pitch'].append(a1)
        angles['roll'].append(a2)
        position['x'].append(px)
        position['y'].append(py)
        position['z'].append(pz)
    return angles, position


def plot_cam_mvts(date, angles, position, dates_interp, angles_interp, position_interp, outdir_cam_mvts):

    fig, ax = plt.subplots(3, 2, figsize=(18, 8), sharex=True, tight_layout=True)
    ax[0, 0].plot(date, angles['yaw'], label = 'yaw (°)')
    ax[0, 0].plot(dates_interp, angles_interp['yaw'], color='r', label = 'yaw interp (°)')
    ax[0, 0].legend(loc='upper right', fontsize=14)
    ax[0, 0].grid(True)
    ax[1, 0].plot(date, angles['pitch'], label = 'pitch (°)')
    ax[1, 0].plot(dates_interp, angles_interp['pitch'], color='r', label = 'pitch interp (°)')
    ax[1, 0].legend(loc='upper right', fontsize=14)
    ax[1, 0].grid(True)
    ax[2, 0].plot(date, angles['roll'], label = 'roll (°)')
    ax[2, 0].plot(dates_interp, angles_interp['roll'], color='r', label='roll interp (°)')
    ax[2, 0].legend(loc='upper right', fontsize=14)
    ax[2, 0].grid(True)
    ax[0, 1].plot(date, position['x'], label = 'x (m)')
    ax[0, 1].legend(loc='upper right', fontsize=14)
    ax[0, 1].grid(True)
    ax[1, 1].plot(date, position['y'], label = 'y (m)')
    ax[1, 1].legend(loc='upper right', fontsize=14)
    ax[1, 1].grid(True)
    ax[2, 1].plot(date, position['z'], label = 'z (m)')
    ax[2, 1].legend(loc='upper right', fontsize=14)
    ax[2, 1].grid(True)
    ax[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    fig.suptitle('CAMERA POSITION IN BEACHCAM COORDINATE SYSTEM')
    fig.autofmt_xdate()
    fig.savefig(outdir_cam_mvts / 'camera_movements.jpg')


def run(dir_h, dir_imgs, ref_img_fn, f_gcps, f_cam_params, dir_gcps, odir_cparams_upd, odir_cparams_upd_smooth,
        outdir_cam_mvts):

    # compute georef parameters for each target image
    date, georef_params_upd = compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, dir_imgs, ref_img_fn, dir_gcps,
                                                  odir_cparams_upd)

    # interp extrinsic parameters of target images
    dates_interp, georef_params_interp = interp_targets_extrinsic(date, georef_params_upd, f_cam_params)

    # compute camera movements
    angles, position = compute_cam_mvts(georef_params_upd)

    # compute camera movements interp
    angles_interp, position_interp = compute_cam_mvts(georef_params_interp)

    # plot camera movements
    plot_cam_mvts(date, angles, position, dates_interp, angles_interp, position_interp, outdir_cam_mvts)
