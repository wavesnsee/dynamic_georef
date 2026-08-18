# Dynamic georef

# rm duplicate of opencv with opencv-python-headless, otherwise there is a qt conflict if user wants a gui or debug.
uv pip uninstall opencv-python
uv pip install --force-reinstall opencv-python-headless==5.0.0.93

in uv.lock, for package roi-editor,  remove dependency opencv-python
