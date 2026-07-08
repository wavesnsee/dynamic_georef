import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from georef.operators import Georef
from dyn_geo.core import img
from scipy.spatial.transform import Rotation


def camera_angles(rvec, tvec):
    """camera_ang: euler angles derived from rotation matrix
    camera_ang[0] = x axis rotation (Pitch-Tilt)
    camera_ang[1] = y axis rotation (Yaw-Azimuth)
    camera_ang[2] = z axis rotation (Roll)

    see opencv doc for camera position comprehension"""

    rmat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rmat, tvec))
    _, _, _, _, _, _, camera_ang = cv2.decomposeProjectionMatrix(P)
    return camera_ang

def camera_position(rotation_mat, translation_mat):
    """
    Compute the camera position from the extrinsic parameters

    Parameters
    ----------
    rotation_mat : numpy array of size (3,1)
            The rotation matrix: extrinsec camera parameters (from the
            calibration function).
    translation_mat : numpy array of size (3,1)
            The translation vector: extrinsic camera parameters (
            from the calibration function)

    Returns
    -------
    cam_posisiton : np.array
        size 3 (Easting_UTM, Northing_UTM, Altitude)
    """
    (rotation, jacobian) = cv2.Rodrigues(rotation_mat)
    cam_position = np.dot(-rotation.transpose(), translation_mat)
    return cam_position

def rotate_vector(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    return data.dot(rotation_matrix)


def run(dir_h, f_gcps, outdir_cam_mvts):

    # read camera georef parameters to get later camera matrix and distorion
    f_camera_parameters = '/home/florent/dev/wavecams/scripts/adjust_camera_pose/camera_parameters_cam44.json'
    georef_params = Georef.from_param_file(f_camera_parameters)

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

    # initialize variables
    angle_0 = []
    angle_1 = []
    angle_2 = []
    t0 = []
    t1 = []
    t2 = []
    date = []

    # loop through homographies
    for f_h in ls_h:
        print(f_h)

        # load homography matrix
        H = np.load(f_h)

        # reverse H
        H = np.linalg.inv(H)

        # apply homography to gcps
        gcps_uv_warped = cv2.perspectiveTransform(gpcs_uv.reshape(-1, 1, 2), H)
        gcps_uv_warped = gcps_uv_warped.reshape(-1, 2)
        gcps_uv_warped = gcps_uv_warped.reshape(gcps_uv_warped.shape[0], 1, gcps_uv_warped.shape[1])
        # plt.plot(gpcs_uv[:,0], gpcs_uv[:, 1], 'k+')
        # plt.plot(projected[:,0], projected[:, 1], 'bd')
        # plt.show()

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

        # image date
        t = img.get_date(f_h)
        date.append(t)

        # get dynamic camera  angles and position
        R, _ = cv2.Rodrigues(rvec)
        R_world = R.T  # Invert: csamera-to-world
        # Choose your convention: 'xyz', 'zyx', 'zxy', etc.
        euler = Rotation.from_matrix(R_world).as_euler('xyz', degrees=True)
        # yaw, pitch, roll = euler
        # print(f"Roll:  {roll:.2f}°")
        # print(f"Pitch: {pitch:.2f}°")
        # print(f"Yaw:   {yaw:.2f}°")

        [p0, p1, p2] = camera_position(rvec, tvec)
        [a0, a1, a2] = camera_angles(rvec, tvec)
        angle_0.append(a0)
        angle_1.append(a1)
        angle_2.append(a2)
        t0.append(p0)
        t1.append(p1)
        t2.append(p2)


    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(date, angle_0)
    ax[1, 0].plot(date, angle_1)
    ax[2, 0].plot(date, angle_2)
    ax[0, 1].plot(date, t0)
    ax[1, 1].plot(date, t1)
    ax[2, 1].plot(date, t2)
    plt.show()

    return