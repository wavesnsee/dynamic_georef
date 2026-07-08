from dyn_geo.core import camera_movements
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    camera_movements.run(path.h,
                         conf.target_imgs.dir,
                         conf.ref_img.fname,
                         conf.ref_img.f_gcps,
                         conf.f_cam_params,
                         path.gcps,
                         path.cam_params_upd_raw,
                         path.cam_params_upd_smooth,
                         path.cam_mvts
                         )