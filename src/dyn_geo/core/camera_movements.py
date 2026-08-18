import io
import json
from copy import copy
from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import pandas as pd
from bokeh.layouts import column, row, gridplot
from bokeh.models import Range1d, ColumnDataSource, Div, CustomJS, Slider, ColorBar
from bokeh.plotting import figure, save, output_file
from bokeh.transform import transform
from georef.operators import Georef, ExtrinsicMatrix
from georef.plot_tools import make_ref_frame, camera_3d_vecs
from scipy.spatial.transform import Rotation as R

from topo_an.core.plot import get_color_mapper
from dyn_geo.core import img
from dyn_geo.core.lidar import lidar_geo2pix
from dyn_geo.core.projection import project_ls_im
from dyn_geo.core.camera_extrinsics import read_cam_params


def smooth_quats(quats, df_periods, smooth_w):

    # initialization
    smoothed_quats = []

    periods = np.unique(df_periods['i_period'])

    for period in periods:

        # get_quaternions in the considered period
        mask = df_periods['i_period'] == period
        quats_p = quats[mask]
        dates_p = df_periods.loc[mask, 'date'].values

        # Force quaternion sign continuity (avoid double-cover flips, q and -q represent the same rotation but flipping sign
        # mid-sequence breaks any component-wise filter)
        for i in range(1, len(quats_p)):
            if np.dot(quats_p[i], quats_p[i - 1]) < 0:
                quats_p[i] *= -1

        # Smooth each quaternion component using a 3-day rolling mean
        df_quat = pd.DataFrame(quats_p, index=dates_p)
        smoothed_quats_p = df_quat.rolling(smooth_w, min_periods=1, center=True).mean().values.copy()

        # renormalize back onto the unit sphere
        smoothed_quats_p /= np.linalg.norm(smoothed_quats_p, axis=1, keepdims=True)

        smoothed_quats.append(smoothed_quats_p)
    
    # concatenate smoothed quaternions
    smoothed_quats = np.concatenate((smoothed_quats))
    
    return smoothed_quats


def smooth_tvecs(georef_params, df_periods, smooth_w):

    # get tvecs from georef_params
    tvecs = []
    for i in range(len(georef_params)):
        tvecs.append(georef_params[i].extrinsic.tvec)
    tvecs = np.array(tvecs)

    # initialization of output smoothed tvec
    smoothed_tvecs = []

    periods = np.unique(df_periods['i_period'])

    for period in periods:
        # get tvecs in the considered period
        mask = df_periods['i_period'] == period
        tvecs_p = tvecs[mask]
        dates_p = df_periods.loc[mask, 'date'].values

        # Smooth each tvec component using a 3-day rolling mean
        df_tvec = pd.DataFrame(tvecs_p, index=dates_p)
        smoothed_tvecs_p = df_tvec.rolling(smooth_w, min_periods=1, center=True).mean().values.copy()

        smoothed_tvecs.append(smoothed_tvecs_p)

    # concatenate smoothed tvecs
    smoothed_tvecs = np.concatenate((smoothed_tvecs))

    return smoothed_tvecs


def smooth_targets_extrinsic(quats, georef_params, df_periods, smooth_w):
    '''
    Smoothing of quaternions and translation vectors
    '''

    # smoothing quaternions
    smoothed_quats = smooth_quats(quats, df_periods, smooth_w)

    # convert smoothed quaternions to rvec
    smoothed_rvecs = R.from_quat(smoothed_quats).as_rotvec()

    # smoothing tvec
    smoothed_tvecs = smooth_tvecs(georef_params, df_periods, smooth_w)

    # initialize output georef_params_smooth, as a list copy of initial georef_params
    georef_params_smooth = [copy(georef_params[0]) for _ in range(len(georef_params))]

    # save smoothed georef parameters in the dedicated variable 'georef_params_smooth'
    for i in range(len(georef_params)):
        extrinsic = ExtrinsicMatrix(smoothed_rvecs[i], smoothed_tvecs[i])
        georef_params_smooth[i].extrinsic = extrinsic

    return georef_params_smooth


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
        try:
            a0, a1, a2 = georef_params.extrinsic.beachcam_angles
            px, py, pz = georef_params.extrinsic.camera_position
        except numpy.linalg.LinAlgError:
            a0, a1, a2, px, py, pz = np.nan * np.ones(6)
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

    return d, valid


def keep_valid(date, georef_params, angles, position, valid, outdir_cam_mvts):
    date = np.array(date)[valid]
    georef_params = np.array(georef_params)[valid]
    angles['pitch'] = np.array(angles['pitch'])[valid]
    angles['yaw'] = np.array(angles['yaw'])[valid]
    angles['roll'] = np.array(angles['roll'])[valid]
    position['x'] = np.array(position['x'])[valid]
    position['y'] = np.array(position['y'])[valid]
    position['z'] = np.array(position['z'])[valid]
    return date, georef_params, angles, position


def plot_despiking(date, position, valid, position_init, d_pos, threshold_d, outdir_cam_mvts):

    def make_plot(date, position, valid, position_init, label):
        x_range = Range1d(min(date), max(date))
        y_range = Range1d(position_init[label][0] - 1, position_init[label][0] + 1)
        p = figure(height=200, x_axis_type='datetime', title=f'Camera position, {label}', x_range=x_range,
                   y_range=y_range, tools="pan,wheel_zoom,box_zoom,reset,save", sizing_mode='stretch_width',
        )

        p.grid.visible = True

        # init line (dashed gold horizontal line)
        p.line(
            x=[date[0], date[-1]], y=[position_init[label], position_init[label]],
            line_width=2, color='gold', line_dash=(4, 4),
            legend_label=f'{label} init (m))'
        )

        # plot camera position
        p.scatter(date, position[label], legend_label=label, color='red', size=10, alpha=1)
        p.scatter(np.array(date)[valid], np.array(position[label])[valid],
                  legend_label=f"{label} valid", color='green', size=10, alpha=1)
        p.yaxis.axis_label = f'{label}'

        return p, x_range

    # Create a global title using a Div
    global_title = Div(text="<h1>Despiking camera movements from camera position in beachcam coordinates system</h1>",
                       sizing_mode='stretch_width')

    # camera position plots
    p1, x_range = make_plot(date, position, valid, position_init, 'x')
    p2, _ = make_plot(date, position, valid, position_init, 'y')
    p3, _ = make_plot(date, position, valid, position_init, 'z')

    # camera diff pos
    p4 = figure(height=200, x_axis_type='datetime', title=f'Difference Camera position (m)',
        x_range=x_range, y_range=Range1d(0, 1),  sizing_mode='stretch_width')
    p4.grid.visible = True
    p4.scatter(date, d_pos, legend_label='diff(m)', color='red', size=10, alpha=1)
    p4.scatter(np.array(date)[valid], np.array(d_pos)[valid], legend_label=f"valid", color='green', size=10, alpha=1)
    # threshold line
    p4.line(
        x=[date[0], date[-1]], y=[threshold_d, threshold_d],
        line_width=2, color='darkred', line_dash=(4, 4), legend_label=f'threshold despiking (m))'
    )

    # save plot
    output_file(outdir_cam_mvts / 'despiking.html', title='DESPIKING')
    layout = column(global_title, p1, p2, p3, p4, sizing_mode='stretch_width')
    save(layout)

    return


def plot_3d_vecs(georef_params, colors=['k', 'b'], axis_names=["x", "y", "z"], title=""):

    # divs = []
    svg_strings = []

    # Make reference frame ready for plot (Ox, Oy, Oz)
    unit_vectors = make_ref_frame()
    camera_c = camera_3d_vecs()

    for i in range(len(georef_params)):
        camera_w = georef_params[i].extrinsic.inv() @ camera_c
        camera_frame_w = georef_params[i].extrinsic.inv() @ (unit_vectors * 0.8)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for j, var in enumerate([camera_w, camera_frame_w]):
            n_vecs = int(var.shape[1] / 2)

            for vec in range(n_vecs):
                start = vec * 2
                end = start + 2
                xs = var[0, start:end]
                ys = var[1, start:end]
                zs = var[2, start:end]
                ax.plot3D(xs, ys, zs, color=colors[j])

                if j == 1:
                    labels = [f"${axis_name}_" + "{" + title + "}$" for axis_name in axis_names]
                    direction = [(ax[1] - ax[0]) for ax in [xs, ys, zs]]
                    position = [x[0] + ((x[1] - x[0]) / 1.1) for x in [xs, ys, zs]]
                    ax.text(*position, labels[vec], direction)

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.grid(False)

        # Render it to an in-memory SVG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        svg_string = buf.getvalue().decode('utf-8')
        svg_strings.append(svg_string)

    return svg_strings


def plot_cam_mvts(date, angles, position, angles_smooth, position_smooth,
                  angles_init, position_init, outdir_cam_mvts):

    output_file(outdir_cam_mvts / 'camera_movements.html', title='CAMERA MOVEMENTS')

    # Create a global title using a Div
    global_title = Div(text="<h1>Camera movements in beachcam coordinate system</h1>", sizing_mode='stretch_width')

    def make_plot(label, unit, raw_vals, smoothed_vals, init_val, x_range=None, y_range=None, title=None):
        p = figure(
            height=260,
            x_axis_type='datetime',
            title=title,
            x_range=x_range,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode='stretch_width',
        )
        if y_range is not None:
            p.y_range.start = y_range[0]
            p.y_range.end = y_range[1]
        p.grid.visible = True

        # init line (dashed gold horizontal line)
        p.line(
            x=[date[0], date[-1]], y=[init_val, init_val],
            line_width=2, color='gold', line_dash=(4, 4),
            legend_label=f'{label} init ({unit})'
        )

        # raw data
        p.line(date, raw_vals, color='gray', legend_label=f'{label} ({unit})')
        p.scatter(date, raw_vals, color='gray', size=6, marker='circle')

        # smoothed data
        p.line(date, smoothed_vals, color='green', legend_label=f'{label} smooth ({unit})')
        p.scatter(date, smoothed_vals, size=4, color='green', marker='circle')

        p.legend.location = 'top_right'
        p.legend.label_text_font_size = '10pt'
        p.legend.click_policy = 'hide'

        return p

    # set x range
    x_range = Range1d(min(date), max(date))

    # plot camera angles and position
    p_yaw = make_plot('yaw', '°', angles['yaw'], angles_smooth['yaw'], angles_init['yaw'], x_range=x_range, title="Camera yaw")
    p_pitch = make_plot('pitch', '°', angles['pitch'], angles_smooth['pitch'], angles_init['pitch'], x_range=x_range, title="Camera pitch")
    p_roll = make_plot('roll', '°', angles['roll'], angles_smooth['roll'], angles_init['roll'], x_range=x_range, title="Camera roll")

    p_x = make_plot('x', 'm', position['x'], position_smooth['x'], position_init['x'], x_range=x_range,
                    y_range=[position_init['x'][0]-0.3, position_init['x'][0] + 0.3], title="Camera position, x")
    p_y = make_plot('y', 'm', position['y'], position_smooth['y'], position_init['y'], x_range=x_range,
                    y_range=[position_init['y'][0]-0.3, position_init['y'][0] + 0.3], title="Camera position, y")
    p_z = make_plot('z', 'm', position['z'], position_smooth['z'], position_init['z'], x_range=x_range,
                    y_range=[position_init['z'][0]-0.3, position_init['z'][0] + 0.3], title="Camera position, z")

    grid = gridplot(
        [[p_yaw, p_x],
         [p_pitch, p_y],
         [p_roll, p_z]],
        toolbar_location='above',
        sizing_mode='stretch_width',
    )
    layout = column(global_title, grid, sizing_mode='stretch_width')
    save(layout)


def save_smooth_cam_params(f_cam_params, date, angles_smooth, position_smooth, odir_cparams_smooth):

    # Read initial camara_parameters file
    with open(f_cam_params, 'r') as f:
        cam_params = json.load(f)

    for i in range(len(date)):

        # compute extrinsic parameters from origin and beachcam angles
        extr = ExtrinsicMatrix.from_origin_beachcam_angles([position_smooth['x'][i], position_smooth['y'][i], position_smooth['z'][i]],
                                                           [angles_smooth['yaw'][i], angles_smooth['pitch'][i], angles_smooth['roll'][i]])

        # save updated camera parameters, changing only extrinsic parameters
        cam_params['extrinsic_parameters']['rvec'] = extr.rvec.reshape(-1).tolist()
        cam_params['extrinsic_parameters']['tvec'] = extr.tvec.reshape(-1).tolist()
        with open(odir_cparams_smooth / f'camera_parameters_{date[i].strftime('%Y%m%d_%H_%M')}.json', 'w') as f:
            json.dump(cam_params, f, indent=2)


def gcps_geo_2pix(f_gcps, georef_params, scaling_percent):

    # read gcps file
    df = pd.read_csv(f_gcps)

    # compute gcps geo coordinates in local srs
    gcps_xyz = df[['easting', 'northing', 'elevation']].to_numpy().T
    gcps_xyz = (georef_params[0].local_srs.m_l_w @ gcps_xyz).T[:, 0:3]
    # plt.plot(gcps_xyz[:, 0], gcps_xyz[:, 1], '+b')
    # plt.show()

    # initialize output variables
    u = []
    v = []
    z = []

    # loop through georef parameters
    for i in range(len(georef_params)):

        # get gcps uv coordinates gor each georef parameter
        uv, valid_pts = georef_params[i].geo2pix(gcps_xyz[:, 0:3])

        # adapt uv to scaling factor
        uv = uv * scaling_percent / 100

        u.append(uv[0][valid_pts])
        v.append(uv[1][valid_pts])
        z.append(gcps_xyz[:, 2][valid_pts])

    return u, v, z


def get_quaternions(georef_params):

    # store targets rvec to a list
    rvecs = []
    for i in range(len(georef_params)):
        rvecs.append(georef_params[i].extrinsic.rvec.squeeze())

    # Initialize the multiple rotations in one Rotation object
    rotations = R.from_rotvec(rvecs)

    # create quaternions from rvecs
    quats = rotations.as_quat()  # (N, 4), scalar-last

    return quats


def get_periods(date, quats):

    def angular_distance(q1, q2):
        """Geodesic distance between two unit quaternions, in radians."""
        dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)  # abs handles double-cover
        return 2 * np.arccos(dot)

    def detect_breakpoints_threshold(motion, k=5.0, min_gap=3):
        median = np.median(motion)
        mad = np.median(np.abs(motion - median)) + 1e-9
        threshold = median + k * mad
        breakpoints = np.where(motion > threshold)[0]

        # merge breakpoints that are too close together (avoid tiny slivers)
        merged_brkpts = []
        for b in breakpoints:
            if not merged_brkpts or b - merged_brkpts[-1] > min_gap:
                merged_brkpts.append(b + 1) # +1 because motion is defined from 1 to n, whereas date is from 0 to to n
        return merged_brkpts

    # compute successive angular distances (in radians)
    ang_d = []
    n = len(quats)
    for i in range(1, n):
        ang_d.append(angular_distance(quats[i], quats[i - 1]))

    # detect indices of breakpoints
    breakpts = detect_breakpoints_threshold(ang_d)

    # compute sub periods
    i_period = []

    # 1st period
    period_0 = date[date < date[breakpts[0]]]
    if len(period_0) > 0:
        i_period.append(np.ones(len(period_0)) * 0)
    # intermediate periods
    for i in range(len(breakpts) - 1):
        period_i = date[np.logical_and(date >= date[breakpts[i]], date < date[breakpts[i + 1]])]
        i_period.append(np.ones(len(period_i)) * (i + 1))
    # last period
    period_last = date[date >= date[breakpts[-1]]]
    if len(period_last) > 0:
        i_period.append(np.ones(len(period_last)) * len(breakpts))
    i_period = np.concatenate((i_period))

    df_period = pd.DataFrame({'date':date, 'i_period':i_period})

    # plot breakpoints
    #fig, ax = plt.subplots()
    #ax.plot(ang_d)
    #for breakpoint in breakpts:
    #    ax.vlines(breakpoint, np.min(ang_d), np.max(ang_d), color='r')
    #plt.show()

    return df_period


def plot_cam_mvts_3d(odir_cparams_smooth, dir_imgs, odir_cam_mvts, f_gcps, pgrid, f_lidar, roi_lidar,
                     start, end, only_at_noon, scaling_pcent=20):
    '''
    # Slider plot of 3D camera movements, and raw/projected images
    '''

    # load georef parameters for each target image
    t_cparams, georef_params = read_cam_params(odir_cparams_smooth, start, end)

    # Filter to keep only the entry closest to noon each day
    if only_at_noon:
        # Convert to pandas Series for easy groupby operations
        ts = pd.Series(range(len(t_cparams)), index=t_cparams)

        # Compute seconds from midnight for each timestamp
        seconds_from_midnight = ts.index.map(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

        # Compute absolute difference from noon (12:00 = 43200 seconds)
        diff_from_noon = pd.Series(np.abs(seconds_from_midnight - 43200), index=ts.index)

        # For each day, find the index with the smallest difference from noon
        noon_indices = diff_from_noon.groupby(ts.index.date).idxmin()

        # Convert timestamp indices back to integer positions for list indexing
        noon_indices = [ts.index.get_loc(idx) for idx in noon_indices]

        # Filter t_cparams and georef_params
        t_cparams = [t_cparams[i] for i in noon_indices]
        georef_params = [georef_params[i] for i in noon_indices]

    # list of uv lidar
    u, v, z = lidar_geo2pix(f_lidar, roi_lidar, odir_cam_mvts, georef_params, scaling_pcent)

    # list of uv gcps
    u_gcps, v_gcps, z_gcps = gcps_geo_2pix(f_gcps, georef_params, scaling_pcent)

    # 3D camera plots
    svg_strings_c3d = plot_3d_vecs(georef_params)

    # list of target images
    ls = sorted(dir_imgs.glob('*.jp*g'))
    t_im = [img.get_date(f) for f in ls]

    # keep only images at dates of camera params
    indices = [i for i, val in enumerate(t_im) if val in t_cparams]
    ls = [ls[i] for i in indices]
    t_im = [t_im[i] for i in indices]

    # list of im dates
    t_im = [t.strftime('%Y-%m-%d %H:%M') for t in t_im]

    # get list of rgba ims
    rgba, width, height = img.ls_im_2rgba(ls, scaling_pcent)

    # pre-compute flipped v-coordinates for lidar display (Bokeh y-axis is top-down)
    v_flipped = [height - v[i] for i in range(len(v))]
    v_flipped_gcps = [height - v_gcps[i] for i in range(len(v_gcps))]

    # Left panel (single Div)
    div = Div(
        text=svg_strings_c3d[0],
        width=400,
        height=400,
    )

    # color mapper
    color_mapper = get_color_mapper(low=np.nanmin(z[0]), high=np.nanmax(z[0]), type='topo')

    # Right panel
    source_im = ColumnDataSource(data=dict(image=[rgba[0]]))

    # raw image
    p = figure(width=width, height=height, x_range=(0, width), y_range=(0, height), title='raw:')
    p.image_rgba(
        image="image",
        source=source_im,
        x=0,
        y=0,
        dw=width,
        dh=height
    )

    source_lidar = ColumnDataSource(data=dict(x=u[0], y=v_flipped[0], z=z[0]))
    source_gcps = ColumnDataSource(data=dict(x=u_gcps[0], y=v_flipped_gcps[0], z=z_gcps[0]))

    p.scatter(
        "x",
        "y",
        source=source_lidar,
        size=3,
        color=transform("z", color_mapper),
        legend_label="lidar"
    )
    p.scatter(
        "x",
        "y",
        source=source_gcps,
        size=4,
        color=transform("z", color_mapper),
        line_color="black",
        line_width=1,
        legend_label="gcps"
    )
    color_bar = ColorBar(color_mapper=color_mapper, title='Elevation (mIGN69)')
    p.add_layout(color_bar, "right")
    p.legend.click_policy = "hide"  # click a legend entry to toggle that series
    p.legend.location = "top_left"

    # project images
    imgs_proj = project_ls_im(ls, georef_params, pgrid, z_proj=9.0)
    # height, width
    w = imgs_proj[0].shape[1]
    h = imgs_proj[0].shape[0]
    # convert to rgba
    imgs_proj = [img.to_rgba(im, h, w) for im in imgs_proj]
    f_zoom = 3
    p2 = figure(width=w * f_zoom, height=h * f_zoom, title='projected:')
    source_im_proj = ColumnDataSource(data=dict(image=[imgs_proj[0]]))
    p2.image_rgba(
        image="image",
        source=source_im_proj,
        x=0,
        y=0,
        dw=width,
        dh=height
    )
    # Remove grid lines
    p2.xgrid.grid_line_color = None
    p2.ygrid.grid_line_color = None
    p2.outline_line_color = None

    slider = Slider(
        start=0,
        end=len(ls) - 1,
        value=0,
        step=1,
        show_value=False
    )

    slider_label = Div(text=f'<b>{t_im[0]}<b>', width=400)

    callback = CustomJS(
        args=dict(
            div=div,
            source_im=source_im,
            source_im_proj=source_im_proj,
            source_lidar=source_lidar,
            source_gcps=source_gcps,
            svgs=svg_strings_c3d,
            imgs=rgba,
            u=u,
            v_flipped=v_flipped,
            z=z,
            u_gcps=u_gcps,
            v_flipped_gcps=v_flipped_gcps,
            z_gcps=z_gcps,
            imgs_proj=imgs_proj,
            t_im=t_im,
            slider_label=slider_label
        ),
        code="""
            const i = cb_obj.value;

            // Update SVG
            div.text = svgs[i];

            // Update image
            source_im.data = {
                image: [imgs[i]]
            };

            // Update scatter lidar
            source_lidar.data = {
                x: u[i],
                y: v_flipped[i],
                z: z[i]
            };

            // Update scatter gcps
            source_gcps.data = {
                x: u_gcps[i],
                y: v_flipped_gcps[i],
                z: z_gcps[i]
            };

            // Update image
            source_im_proj.data = {
                image: [imgs_proj[i]]
            };

            source_lidar.change.emit();
            slider_label.text = "<b>" + t_im[i] + "</b>";
        """,
    )


    slider.js_on_change("value", callback)

    # Hide ticks, labels, axis line
    p.xaxis.visible = False
    p.yaxis.visible = False
    p2.xaxis.visible = False
    p2.yaxis.visible = False

    layout = column(
        row(column(div, column(slider, slider_label)), column(p, p2))
    )

    output_file(odir_cam_mvts / 'camera_movements_3d.html', title='3D CAM MOVEMENTS')
    save(layout)

    return


def run(f_cam_params, dir_cparams_raw, odir_cparams_smooth, odir_cam_mvts, smooth_w):

    # compute camera position from initial georef
    angles_init, position_init = compute_cam_mvts([Georef.from_param_file(f_cam_params)])

    # load georef parameters for each target image
    date, georef_params = read_cam_params(dir_cparams_raw)

    # compute camera movements of each target image
    angles, position = compute_cam_mvts(georef_params)

    # Despike camera movements
    threshold_d = 0.25 # in cm
    d_pos, valid = despike_cam_mvts(position_init, position, threshold_d=threshold_d)

    # plot despiking
    plot_despiking(date, position, valid, position_init, d_pos, threshold_d, odir_cam_mvts)

    # keep only valid data
    date, georef_params, angles, position = keep_valid(date, georef_params, angles, position, valid, odir_cam_mvts)

    # quaternions
    quats = get_quaternions(georef_params)

    # get subperiods from large camera movements
    df_periods = get_periods(date, quats)

    # smooth extrinsic parameters of target images
    georef_params_smooth = smooth_targets_extrinsic(quats, georef_params, df_periods, smooth_w)

    # compute smoothed camera movements
    angles_smooth, position_smooth = compute_cam_mvts(georef_params_smooth)

    # plot camera movements raw and smoothed
    plot_cam_mvts(date, angles, position,
                  angles_smooth, position_smooth,
                  angles_init, position_init,
                  odir_cam_mvts)

    # compute and save smoothed camera parameters
    save_smooth_cam_params(f_cam_params, date, angles_smooth, position_smooth, odir_cparams_smooth)