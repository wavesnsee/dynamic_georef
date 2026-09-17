from dyn_geo.core import accuracy_metrics
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    accuracy_metrics.run(path.matches_data_filtered,
                         path.h,
                         path.acc_metrics,
                         conf.ref_img.fname,
                         conf.target_imgs.dir,
                         conf.f_cam_params,
    )
