from stab_fm.core import accuracy_metrics
from stab_fm.cli.paths_subdirs_out import Paths

def main(conf):
    accuracy_metrics.run(
        Paths(conf.outdir, conf.matching)
    )
