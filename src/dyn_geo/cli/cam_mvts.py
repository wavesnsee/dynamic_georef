from dyn_geo.core import camera_movements
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    camera_movements.run(path.h,
                         conf.ref_img.f_gcps,
                         path.cam_mvts
                         )