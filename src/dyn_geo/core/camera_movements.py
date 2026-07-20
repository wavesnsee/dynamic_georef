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
from scipy.interpolate import make_splprep
from datetime import timedelta, datetime
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d, RangeTool, ColumnDataSource, DatetimeTickFormatter
from bokeh.layouts import column, gridplot
from bokeh.models import Div


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


def interp_targets_rvec(georef_params_upd, dates, interp_dates):

    # store targets rvec to a list
    rvecs = []
    for i in range(len(georef_params_upd)):
        rvecs.append(georef_params_upd[i].extrinsic.rvec.squeeze())

    # Initialize the multiple rotations in one Rotation object
    rots = R.from_rotvec(rvecs)

    # Create the interpolator object
    slerp = Slerp(dates, rots)

    # perform Spherical Linear Interpolation of Rotations
    interp_rots = slerp(interp_dates)

    # save interp_rots as rotation vectors
    interp_rvecs = [interp_rots[i].as_rotvec() for i in range(len(interp_rots))]

    return interp_rvecs


def interp_targets_tvec(georef_params_upd, dates, interp_dates):

    # store targets txn ty, tz to lists
    tx = []
    ty = []
    tz = []
    for i in range(len(georef_params_upd)):
        tx_i, ty_i, tz_i = (georef_params_upd[i].extrinsic.tvec.squeeze())
        tx.append(tx_i)
        ty.append(ty_i)
        tz.append(tz_i)

    # normalize dates
    dates = np.array(dates)
    u = (dates - dates.min()) / (dates.max() - dates.min())
    interp_dates = np.array(interp_dates)
    u_interp =  (interp_dates - dates.min()) / (dates.max() - dates.min())

    # Interpolate with smoothing
    spline, u_fit = make_splprep([tx, ty, tz],
                                 u=u,
                                 s=5,  # smoothing parameter
                                 k=3)  # cubic

    tvec_interp = spline(u_interp)

    return tvec_interp


def interp_targets_extrinsic(dates, georef_params_upd, f_cam_params):
    '''
    Interpolation of rotation vector to compute smoothed camera parameters
    '''

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation

    # read georef parameters, that will be updated for each target image
    georef_params = Georef.from_param_file(f_cam_params)

    def round_to_day_at_noon(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=0.5)

    # dates to perform interpolation
    interp_dates = np.arange(round_to_day_at_noon(dates[0]), round_to_day_at_noon(dates[-1]) + timedelta(days=1),
                             timedelta(days=1)).astype(datetime)
    interp_dates = interp_dates[np.logical_and((interp_dates > dates.min()), (interp_dates < dates.max()))]

    # convert dates to num
    dates = [mdates.date2num(dates[i]) for i in range(len(dates))]
    interp_dates = [mdates.date2num(interp_dates[i]) for i in range(len(interp_dates))]

    # initialize list of Georef objects
    georef_params_interp = [copy(georef_params) for _ in range(len(interp_dates))]

    # interpolation of rvec
    interp_rvecs = interp_targets_rvec(georef_params_upd, dates, interp_dates)

    # interpolation of tvec
    interp_tvecs = interp_targets_tvec(georef_params_upd, dates, interp_dates)

    # convert interpolated dates back to datetime
    interp_dates = [mdates.num2date(interp_dates[i]) for i in range(len(interp_dates))]

    # save updated georef parameters
    for i in range(len(interp_rvecs)):
        # if we want to leave tvec unchanged
        # extrinsic_interp = ExtrinsicMatrix(interp_rvecs[i], georef_params_upd[0].extrinsic.tvec)
        # otherwise:
        extrinsic_interp = ExtrinsicMatrix(interp_rvecs[i], interp_tvecs[:, i])
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


def despike_cam_mvts(position_ref, position, threshold_d=0.6):

    # compute target's camera position distance from reference camera position
    d = []
    for i in range(len(position['x'])):
        diff_pos = np.array([
            [position['x'][i] - position_ref['x']],
            [position['y'][i] - position_ref['y']],
            [position['z'][i] - position_ref['z']]
        ]
        )
        d.append(np.sqrt((diff_pos ** 2).sum()))

    valid = np.array(d) < threshold_d

    return valid


def keep_valid(date, georef_params_upd, angles, position, valid):
    date = np.array(date)[valid]
    georef_params_upd = np.array(georef_params_upd)[valid]
    angles['pitch'] = np.array(angles['pitch'])[valid]
    angles['yaw'] = np.array(angles['yaw'])[valid]
    angles['roll'] = np.array(angles['roll'])[valid]
    position['x'] = np.array(position['x'])[valid]
    position['y'] = np.array(position['y'])[valid]
    position['z'] = np.array(position['z'])[valid]
    return date, georef_params_upd, angles, position


def plot_despiking(date, position, valid):
    plot_h = 250
    plot_w = 1800
    x_range = Range1d(min(date), max(date))

    # Create a global title using a Div
    global_title = Div(text="<h1>Despiking camera movements from camera position in beachcam coordinates system</h1>", width=plot_w)

    # plot camera position, x
    p1 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", x_range=x_range)
    p1.scatter(date, position['x'], legend_label="x", color='red', size=10, alpha=1)
    p1.scatter(np.array(date)[valid], np.array(position['x'])[valid], legend_label="x valid", color='green', size=10,
               alpha=1)
    p1.yaxis.axis_label = 'Camera position, x'

    # plot camera position, y
    p2 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", x_range=x_range)
    p2.scatter(date, position['y'], legend_label="y", color='red', size=10, alpha=1)
    p2.scatter(np.array(date)[valid], np.array(position['y'])[valid], legend_label="y valid", color='green', size=10,
               alpha=1)
    p2.yaxis.axis_label = 'Camera position, y'

    # plot camera position, z
    p3 = figure(width=plot_w, height=plot_h, tools="xpan,xwheel_zoom,reset", x_range=x_range)
    p3.scatter(date, position['z'], legend_label="z", color='red', size=10, alpha=1)
    p3.scatter(np.array(date)[valid], np.array(position['z'])[valid], legend_label="z valid", color='green', size=10,
               alpha=1)
    p3.yaxis.axis_label = 'Camera position, z'


    name = 'despiking.html'
    output_file(name)
    layout = column(global_title, p1, p2, p3)
    save(layout)

    return


def plot_cam_mvts(date, angles, position, dates_interp, angles_interp, position_interp,
                   angles_init, position_init, outdir_cam_mvts):

    output_file(outdir_cam_mvts / 'camera_movements.html', title='CAMERA POSITION IN BEACHCAM COORDINATE SYSTEM')

    # Create a global title using a Div
    global_title = Div(text="<h1>Camera movements</h1>", width=500)

    def make_plot(label, unit, raw_vals, interp_vals, init_val, x_range=None, title=None):
        p = figure(
            width=200, height=260,
            x_axis_type='datetime',
            title=title,
            x_range=x_range,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        p.grid.visible = True

        # init line (dashed gold horizontal line)
        p.line(
            x=[date[0], date[-1]], y=[init_val, init_val],
            line_width=2, color='gold', line_dash=(4, 4),
            legend_label=f'{label} init ({unit})'
        )

        # raw data
        p.line(date, raw_vals, legend_label=f'{label} ({unit})')
        p.scatter(date, raw_vals, size=8, marker='circle')

        # interpolated data
        p.line(dates_interp, interp_vals, color='red', legend_label=f'{label} interp ({unit})')
        p.scatter(dates_interp, interp_vals, size=8, color='red', marker='circle')

        p.legend.location = 'top_right'
        p.legend.label_text_font_size = '10pt'
        p.legend.click_policy = 'hide'

        return p

    # set x range
    x_range = Range1d(min(date), max(date))

    # plot camera angles and position
    p_yaw = make_plot('yaw', '°', angles['yaw'], angles_interp['yaw'], angles_init['yaw'], x_range=x_range, title="Camera yaw")
    p_pitch = make_plot('pitch', '°', angles['pitch'], angles_interp['pitch'], angles_init['pitch'], x_range=x_range, title="Camera pitch")
    p_roll = make_plot('roll', '°', angles['roll'], angles_interp['roll'], angles_init['roll'], x_range=x_range, title="Camera roll")

    p_x = make_plot('x', 'm', position['x'], position_interp['x'], position_init['x'], x_range=x_range, title="Camera position, x")
    p_y = make_plot('y', 'm', position['y'], position_interp['y'], position_init['y'], x_range=x_range, title="Camera position, y")
    p_z = make_plot('z', 'm', position['z'], position_interp['z'], position_init['z'], x_range=x_range, title="Camera position, z")

    grid = gridplot(
        [[p_yaw, p_x],
         [p_pitch, p_y],
         [p_roll, p_z]],
        toolbar_location='above',
        sizing_mode='stretch_width',
    )
    layout = column(global_title, grid)
    save(layout)

def run(dir_h, dir_imgs, ref_img_fn, f_gcps, f_cam_params, dir_gcps, odir_cparams_upd, odir_cparams_upd_smooth,
        outdir_cam_mvts):

    # compute camera position of initial georef
    angles_init, position_init = compute_cam_mvts([Georef.from_param_file(f_cam_params)])

    # compute georef parameters for each target image
    date, georef_params_upd = compute_targets_extrinsic(dir_h, f_gcps, f_cam_params, dir_imgs, ref_img_fn, dir_gcps,
                                                  odir_cparams_upd)

    # compute camera movements of each target image
    angles, position = compute_cam_mvts(georef_params_upd)

    # Despike camera movements
    valid = despike_cam_mvts(position_init, position)

    # plot despiking
    plot_despiking(date, position, valid)

    # keep only valid data
    date, georef_params_upd, angles, position = keep_valid(date, georef_params_upd, angles, position, valid)

    # interp extrinsic parameters of target images
    dates_interp, georef_params_interp = interp_targets_extrinsic(date, georef_params_upd, f_cam_params)

    # compute camera movements interp
    angles_interp, position_interp = compute_cam_mvts(georef_params_interp)

    # plot camera movements raw and interpolated
    plot_cam_mvts(date, angles, position,
                  dates_interp, angles_interp, position_interp,
                  angles_init, position_init,
                  outdir_cam_mvts)
