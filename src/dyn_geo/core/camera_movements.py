import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2
import json
from georef.operators import Georef
from dyn_geo.core import img
from scipy.spatial.transform import Rotation


def rotate_vector(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    return data.dot(rotation_matrix)


def compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, outdir_cam_params_upd):

    # Read initial camara_parameters file
    with open(f_cam_params, 'r') as f:
        cam_params = json.load(f)

    # read camera parameters, whose extrinsic params will be updated for each target image
    georef_params = Georef.from_param_file(f_cam_params)

    # read gcps file
    df = pd.read_csv(f_gcps)

    # extract gcps pixel coordinates
    gpcs_uv = df[['U', 'V']].to_numpy()

    # extract gcps geo coordinates
    gcps_xy = df[['easting', 'northing']].to_numpy()
    gcps_xyz = df[['easting', 'northing', 'elevation']].to_numpy()

    ## applying rotation to the local system
    data = np.stack((gcps_xy[:, 0] - georef_params.local_srs.offset_easting,
                     gcps_xy[:, 1] - georef_params.local_srs.offset_northing)).T
    rotated_data = rotate_vector(data, georef_params.local_srs.rotation)

    gcps_xyz[:, 0] = rotated_data[:, 0]
    gcps_xyz[:, 1] = rotated_data[:, 1]
    gcps_xyz = gcps_xyz.astype(np.float32)

    gcps_xyz = gcps_xyz.reshape(gcps_xyz.shape[0], 1, gcps_xyz.shape[1])

    # list of homography matrixes
    ls_h = sorted(dir_h.glob('*.npy'))

    # loop through homographies
    for f_h in ls_h:
        # load homography matrix
        H = np.load(f_h)

        # reverse H
        H = np.linalg.inv(H)

        # apply homography to gcps
        gcps_uv_warped = cv2.perspectiveTransform(gpcs_uv.reshape(-1, 1, 2), H)
        gcps_uv_warped = gcps_uv_warped.reshape(-1, 2)
        gcps_uv_warped = gcps_uv_warped.reshape(gcps_uv_warped.shape[0], 1, gcps_uv_warped.shape[1])

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

        # save updated camera parameters, changing only extrinsic parameters
        cam_params['extrinsic_parameters']['rvec'] = rvec.reshape(-1).tolist()
        cam_params['extrinsic_parameters']['tvec'] = tvec.reshape(-1).tolist()
        with open(outdir_cam_params_upd / f_h.name.replace('.npy', '.json'), 'w') as f:
            json.dump(cam_params, f, indent=2)
    return


def compute_cam_mvts(outdir_cam_params_upd):
    ls = sorted(outdir_cam_params_upd.glob('*.json'))

    date = []
    angles = {}
    position = {}
    angles[0] = []
    angles[1] = []
    angles[2] = []
    position[0] = []
    position[1] = []
    position[2] = []

    for f_cp in ls:
        # read individual camera parameters
        georef_params = Georef.from_param_file(f_cp)

        # time
        t = img.get_date(f_cp)
        date.append(t)

        # get camera angles and position
        a0, a1, a2 = georef_params.extrinsic.camera_angles
        p0, p1, p2 = georef_params.extrinsic.camera_position
        angles[0].append(a0)
        angles[1].append(a1)
        angles[2].append(a2)
        position[0].append(p0)
        position[1].append(p1)
        position[2].append(p2)
    return date, angles, position


def plot_cam_mvts(date, angles, position, outdir_cam_mvts):

    fig, ax = plt.subplots(3, 2, figsize=(18, 8), sharex=True, tight_layout=True)
    ax[0, 0].plot(date, angles[0], label = 'angle 0')
    ax[0, 0].legend(loc='upper right', fontsize=14)
    ax[0, 0].grid(True)
    ax[1, 0].plot(date, angles[1], label = 'angle 1')
    ax[1, 0].legend(loc='upper right', fontsize=14)
    ax[1, 0].grid(True)
    ax[2, 0].plot(date, angles[2], label = 'angle 2')
    ax[2, 0].legend(loc='upper right', fontsize=14)
    ax[2, 0].grid(True)
    ax[0, 1].plot(date, position[0], label = 'position 0')
    ax[0, 1].legend(loc='upper right', fontsize=14)
    ax[0, 1].grid(True)
    ax[1, 1].plot(date, position[1], label = 'position 1')
    ax[1, 1].legend(loc='upper right', fontsize=14)
    ax[1, 1].grid(True)
    ax[2, 1].plot(date, position[2], label = 'position 2')
    ax[2, 1].legend(loc='upper right', fontsize=14)
    ax[2, 1].grid(True)
    ax[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    fig.suptitle('CAMERA MOVEMENTS')
    fig.autofmt_xdate()
    fig.savefig(outdir_cam_mvts / 'camera_movements.jpg')


def run(dir_h, f_gcps, f_cam_params, outdir_cam_params_upd, outdir_cam_mvts):

    # compute extrinsic parameters for each target image
    compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, outdir_cam_params_upd)

    # compute camera movements
    date, angles, position= compute_cam_mvts(outdir_cam_params_upd)

    # plot camera movements
    plot_cam_mvts(date, angles, position, outdir_cam_mvts)
