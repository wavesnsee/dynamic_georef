import pandas as pd
import shapely
from roi_editor.core import roi
import numpy as np


def masks_from_rois(f_roi, im_shape):
    '''
    computes masks 2d and union of these masks, from a vector roi file and the shape of the image where they are
    intended to be applied
    '''

    # read rois
    roi_ = roi.ROICollection(im_shape)
    rois = roi_.load_from_json(f_roi)
    rois = rois.rois

    # compute masks from rois
    masks = [rois[i].compute_mask(im_shape).astype(np.uint8)*255 for i in range(len(rois))]

    # compute union of masks
    mask = np.any(masks, axis=0).astype(np.uint8)*255

    return masks, mask


def pts_inside(
    roi_ld: shapely.Polygon,
    df: pd.DataFrame,
    x: str = "U",
    y: str = "V",
) -> np.ndarray:
    """
    Filter points by keeping only the ones inside a polygon.

    Parameters
    ----------
    roi_ld : shapely.Polygon
        Polygon defining the region of interest.
    df : pd.DataFrame
        DataFrame containing the points.
    x : str, default="U"
        Name of the x-coordinate column.
    y : str, default="V"
        Name of the y-coordinate column.

    Returns
    -------
    np.ndarray
        Boolean mask, with one value per row of `df`.
    """
    points = shapely.points(df[x].to_numpy(), df[y].to_numpy())

    return shapely.covers(roi_ld, points)