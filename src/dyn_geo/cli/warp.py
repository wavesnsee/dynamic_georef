from dyn_geo.core import warp_imgs
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    warp_imgs.run(conf.target_imgs.dir,
                  path.h,
                  path.warped
                  )

