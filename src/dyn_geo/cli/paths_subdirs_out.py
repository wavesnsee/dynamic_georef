from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    outdir: Path
    matching_technique: 'str'
    matches: Path = field(init=False)
    matches_plot: Path = field(init=False)
    matches_data: Path = field(init=False)
    acc_metrics: Path = field(init=False)
    cam_mvts: Path = field(init=False)
    cam_params_raw: Path = field(init=False)
    cam_params_smooth: Path = field(init=False)
    h: Path = field(init=False)
    gcps: Path = field(init=False)
    warped: Path = field(init=False)

    def __post_init__(self):
        """Initialize subdirectories after creation"""
        self.outdir = self.outdir / self.matching_technique
        self.matches = self.outdir / "matches"
        self.matches_plot = self.matches / "plots"
        self.matches_data = self.matches / "data"
        self.acc_metrics = self.outdir / "acc_metrics"
        self.cam_mvts = self.outdir / "cam_mvts"
        self.cam_params_raw = self.outdir / "cam_params" / "raw"
        self.cam_params_smooth = self.outdir / "cam_params" / "smooth"
        self.h = self.outdir / "H"
        self.gcps  = self.outdir / "gcps"
        self.warped = self.outdir / "warped"
        self.create_all()

    def create_all(self):
        """Create all directories"""
        for path in [self.matches_plot, self.matches_data, self.h, self.acc_metrics, self.cam_mvts,
                     self.cam_params_raw, self.cam_params_smooth, self.gcps, self.warped]:
            path.mkdir(parents=True, exist_ok=True)
        return self
