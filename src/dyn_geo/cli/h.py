from dyn_geo.core import homography
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    homography.run(Paths(conf.outdir, conf.matching), conf.roi_low_distort)
