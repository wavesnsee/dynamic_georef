from stab_fm.core import feature_matching
from stab_fm.cli.paths_subdirs_out import Paths

def main(conf):

    feature_matching.run(
        conf.ref_img.fname,
        conf.ref_img.f_rois_fm,
        conf.target_imgs.dir,
        conf.f_calib,
        conf.matching,
        Paths(conf.outdir, conf.matching)
    )


def plot(conf):

    paths = Paths(conf.outdir, conf.matching)
    feature_matching.plot(conf.ref_img.fname,
                          conf.target_imgs.dir,
                          paths.matches_data,
                          paths.matches_plot)
