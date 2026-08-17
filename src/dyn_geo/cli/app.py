import sys
from pathlib import Path
import traceback
from typing import Annotated
from datetime import datetime
import yaml
from pydantic import BaseModel

import typer

from dyn_geo.cli import fm, accuracy, warp, cam_mvts

app = typer.Typer(no_args_is_help=True)


class RefImg(BaseModel):
    fname: Path
    f_rois_fm: Path
    f_rois_edges: Path
    f_gcps: Path

class TargetImgs(BaseModel):
    dir: Path

class ProjectionGrid(BaseModel):
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    res: float
    z: float

class Plot3d(BaseModel):
    f_lidar: Path
    roi_lidar: Path
    start: datetime
    end: datetime

class AppConfig(BaseModel):
    ref_img: RefImg
    target_imgs: TargetImgs
    f_cam_params: Path
    matching: str
    pgrid: ProjectionGrid
    start: datetime
    end: datetime
    smooth_w: str
    outdir: Path
    plot3d: Plot3d
    compute_fm: bool
    plot_fm: bool
    acc_metrics: bool
    compute_raw_extrinsic: bool
    compute_smooth_extrinsic: bool
    plot_cam_3d_mvts: bool
    warp: bool


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)  # validation automatique

@app.command()
def main(
    input_yaml: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            help="Input yaml file containing parameters",
        ),
    ],
):
    # load configuration file
    conf = load_config(input_yaml)

    if not conf.ref_img.fname.exists():
        raise typer.Exit("Reference image does not exist")

    try:
        # Run feature matching
        if conf.compute_fm:
            print('run feature matching and save Homography transforms')
            fm.main(conf)

        # Plot feature matching
        if conf.plot_fm:
            print('plot matching points')
            fm.plot(conf)

        # Compute accuracy metrics
        if conf.acc_metrics:
            print('compute accuracy metrics between ref and target images')
            accuracy.main(conf)

        # Compute camera raw extrinsics, and smooth extrinsics
        cam_mvts.main(conf)

        # Plot 3d camera movements
        if conf.plot_cam_3d_mvts:
            print('plot 3D camera movements')
            cam_mvts.plot_3d(conf)

        # warp
        if conf.warp:
            print('warp images')
            warp.main(conf)

    except Exception as e:  # noqa: BLE001
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        typer.echo(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    app()
