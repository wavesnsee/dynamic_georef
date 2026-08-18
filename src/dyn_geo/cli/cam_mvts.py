from dyn_geo.core import camera_extrinsics, camera_movements
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    paths = Paths(conf.outdir, conf.matching)

    # compute camera raw extrinsics
    if conf.compute_raw_extrinsic:
        print('compute camera raw extrinsics')
        camera_extrinsics.run(paths.h,
                             conf.target_imgs.dir,
                             conf.ref_img.fname,
                             conf.ref_img.f_gcps,
                             conf.f_cam_params,
                             conf.start,
                             conf.end,
                             paths.gcps,
                             paths.cam_params_raw)

    # compute camera smooth extrinsics
    if conf.compute_smooth_extrinsic:
        print('compute camera movements and smooth extrinsics')

        camera_movements.run(conf.f_cam_params,
                             paths.cam_params_raw,
                             paths.cam_params_smooth,
                             paths.cam_mvts,
                             conf.smooth_w,
                             conf.cam_id
                             )

def plot_3d(conf):

    paths = Paths(conf.outdir, conf.matching)

    # plot 3D camera movements, in parallel with raw and projected images
    camera_movements.plot_cam_mvts_3d(paths.cam_params_smooth,
                                      conf.target_imgs.dir,
                                      paths.cam_mvts,
                                      conf.ref_img.f_gcps,
                                      conf.pgrid,
                                      conf.plot3d.f_lidar,
                                      conf.plot3d.roi_lidar,
                                      conf.plot3d.start,
                                      conf.plot3d.end,
                                      conf.plot3d.only_at_noon
                                      )

