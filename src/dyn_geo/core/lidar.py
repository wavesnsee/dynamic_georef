import georef.operators
import numpy as np
from pathlib import Path
from typing import List
from topo_an.core.geo_utils import reproject_rasters, raster_grid
from topo_an.core.topo import open_sporadic_topos, apply_roi_mask_to_sporadic_topos


def lidar_2_local(f_lidar: Path, roi_lidar: Path, odir_cam_mvts:Path, georef_params: List[georef.operators.Georef]):
    '''
    read lidar data.
    return altitude data in the local reference system of wns site
    '''
    # read lidar
    lidar = open_sporadic_topos([f_lidar], '2154')

    # apply roi mask to lidar topography
    outdir_masked = odir_cam_mvts / 'lidar'
    lidar = apply_roi_mask_to_sporadic_topos(lidar, roi_lidar, outdir_masked)[0]

    # change crs of lidar to crs of site if necessary
    if lidar.crs.to_epsg() != georef_params[0].local_srs.horizontal_srs.auth_srid:
        z, left, bottom, right, top = reproject_rasters([lidar],
                                                        crs=georef_params[0].local_srs.horizontal_srs.auth_srid,
                                                        flipud_bokeh=False)

    # get lidar grid coordinates
    X, Y = raster_grid(lidar, georef_params[0].local_srs.horizontal_srs.auth_srid)

    # convert lidar points in local coordinate system
    xyz = np.vstack((X.ravel(), Y.ravel(), z[0].ravel())).T
    lidar_srs_local = (georef_params[0].local_srs.m_l_w @ xyz).T
    return z, lidar_srs_local


def get_lidar_uv(f_lidar: Path, roi_lidar: Path, odir_cam_mvts:Path, georef_params: List[georef.operators.Georef],
                 scaling_percent: int):

    # get lidar geo data expressed in the local reference system
    z, lidar_srs_local = lidar_2_local(f_lidar, roi_lidar, odir_cam_mvts, georef_params)
    z = z[0].ravel()

    # initialize output variables (list of uv coordinates, list of valid points)
    u_lidar = []
    v_lidar = []
    valid_points = []

    # loop through georef parameters
    for i in range(len(georef_params)):

        # get lidar uv coordinates gor each georef parameter
        uv, valid_pts = georef_params[i].geo2pix(lidar_srs_local[:, 0:3])

        # adapt uv to scaling factor
        uv = uv * scaling_percent / 100

        u_lidar.append(uv[0])
        v_lidar.append(uv[1])
        valid_points.append(valid_pts)

    u_lidar = [u_lidar[i][valid_points[i]] for i in range(len(u_lidar))]
    v_lidar = [v_lidar[i][valid_points[i]] for i in range(len(v_lidar))]
    z = [z[valid_points[i]] for i in range(len(valid_points))]

    return u_lidar, v_lidar, z




