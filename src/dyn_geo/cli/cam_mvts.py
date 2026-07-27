from dyn_geo.core import camera_extrinsics, camera_movements
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf, compute_raw_extrinsic):

    path = Paths(conf.outdir, conf.matching)

    # compute camera raw extrinsics
    if compute_raw_extrinsic:
        print('compute camera raw extrinsics')
        camera_extrinsics.run(path.h,
                             conf.target_imgs.dir,
                             conf.ref_img.fname,
                             conf.ref_img.f_gcps,
                             conf.f_cam_params,
                             path.gcps,
                             path.cam_params_raw)

    print('compute camera movements and smooth extrinsics')
    camera_movements.run(conf.target_imgs.dir,
                         conf.ref_img.f_gcps,
                         conf.f_cam_params,
                         path.cam_params_raw,
                         path.cam_params_smooth,
                         path.cam_mvts
                         )