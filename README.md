# Dynamic georef

Dynamic_georef is a project dedicated to compute extrinsic parameters of a camera through time, 
based on feature matching detections and ground control points.

## Installation

```bash
uv sync
```

#### remove duplicate of opencv (coming from roi_editor) with opencv-python-headless, otherwise there is a qt conflict if user wants a gui or debug.
```bash
uv pip uninstall opencv-python
uv pip install --force-reinstall opencv-python-headless==5.0.0.93

in uv.lock, for package roi-editor,  remove dependency opencv-python
```
