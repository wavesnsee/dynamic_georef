import sys
from pathlib import Path
import traceback
from typing import Annotated
import yaml
from pydantic import BaseModel

import typer

from stab_fm.cli import *

app = typer.Typer(no_args_is_help=True)


class RefImg(BaseModel):
    fn: Path
    f_rois: Path

class TargetImgs(BaseModel):
    dir: Path

class AppConfig(BaseModel):
    ref_img: RefImg
    target_imgs: TargetImgs
    outdir: Path

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

    if not conf.ref_img.fn.exists():
        raise typer.Exit("Reference image does not exist")

    if not conf.outdir.exists():
        conf.outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Run feature matching
        print('')
        # Compute and apply stabilization transform



    except Exception as e:  # noqa: BLE001
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        typer.echo(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    app()
