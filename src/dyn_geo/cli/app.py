import sys
from pathlib import Path
import traceback
from typing import Annotated
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

class AppConfig(BaseModel):
    ref_img: RefImg
    target_imgs: TargetImgs
    outdir: Path
    f_cam_params: Path
    matching: str
    compute_fm: bool
    plot_fm: bool
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
        print('compute accuracy metrics between ref and target images')
        accuracy.main(conf)

        # Compute camera movements
        print('compute dynamic georefs')
        cam_mvts.main(conf)

        # warp
        if conf.warp:
            warp.main(conf)

    except Exception as e:  # noqa: BLE001
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        typer.echo(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    app()
