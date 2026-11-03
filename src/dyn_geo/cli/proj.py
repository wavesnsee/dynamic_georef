from dyn_geo.core import projection
from dyn_geo.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    projection.run(conf.target_imgs.dir,
                   conf.start,
                   conf.end,
                   conf.pgrid,
                   path.cam_params_raw,
                   path.cam_mvts,
                   path.projected
                  )