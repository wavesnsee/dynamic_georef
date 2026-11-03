from dyn_geo.core import homography
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    homography.compute(path.matches_data_raw,
                       path.h,
                       path.matches_data_filtered,
                       conf.f_roi_low_distort)
