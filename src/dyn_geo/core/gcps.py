from pathlib import Path

import numpy as np
from shapely import Polygon
import pandas as pd
import cv2

from georef.operators import Georef

from dyn_geo.core.mask import read_polygon_roi, pts_inside


def read_gcps(f_gcps: Path,
              roi_ld: Polygon,
              camera_matrix: np.array,
              dist_coeffs: np.array) -> pd.DataFrame:
    '''
    read pixel and geo coordinates of ground control points. Only gcps whose pixel points are inside roi of low
    distorsion are kept
    '''

    # read csv with pandas
    df = pd.read_csv(f_gcps)

    # extract gcps uv (pixel) coordinates
    gcps_uv = df[['U', 'V']].to_numpy()

    # undistort gcps uv
    gcps_uv = cv2.undistortPoints(gcps_uv.reshape(-1, 1, 2), camera_matrix,
                                  dist_coeffs, P=camera_matrix).reshape(-1, 2)
    df['U'] = gcps_uv[:, 0]
    df['V'] = gcps_uv[:, 1]

    # mask of gcps_uv points inside roi of low distorsion
    mask_ld = pts_inside(roi_ld, df, x='U', y='V')

    return df[mask_ld]


def get_gcps_uv(df: pd.DataFrame) -> np.array:
    '''
    get pixel coordinates of ground control points
    '''
    return df[['U', 'V']].to_numpy()


def get_gcps_xyz(df: pd.DataFrame, georef_params: Georef) -> np.array:
    '''
    geo geo coordinates of ground control points, expressed in the local srs, with a format compatible with
    solvePnPRansac
    '''

    # compute gcps geo coordinates in local srs
    gcps_xyz = df[['easting', 'northing', 'elevation']].to_numpy().T
    gcps_xyz = (georef_params.local_srs.m_l_w @ gcps_xyz).T[:, 0:3]

    # reshape gcps_xyz to make it compatible with solvePnPRansac
    gcps_xyz = gcps_xyz.reshape(gcps_xyz.shape[0], 1, gcps_xyz.shape[1])

    return gcps_xyz


def apply_h_to_gcps(f_h: Path,
                    gcps_uv: np.array,
                    df: pd.DataFrame,
                    roi_ld: Polygon):

    # load homography matrix
    H = np.load(f_h)

    # reverse H
    H = np.linalg.inv(H)

    # apply homography to gcps
    gcps_uv_warped = cv2.perspectiveTransform(gcps_uv.reshape(-1, 1, 2), H)
    gcps_uv_warped = gcps_uv_warped.reshape(-1, 2)
    df['U_warped'] = gcps_uv_warped[:, 0]
    df['V_warped'] = gcps_uv_warped[:, 1]

    # mask of gcps_uv_warped points inside roi of low distorsion
    mask_ld = pts_inside(roi_ld, df, x='U_warped', y='V_warped')

    # reshape gcps_uv_warped to make it compatible with solvePnPRansac
    gcps_uv_warped = gcps_uv_warped.reshape(gcps_uv_warped.shape[0], 1, gcps_uv_warped.shape[1])

    return mask_ld, gcps_uv_warped
