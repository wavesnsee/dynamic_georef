from roi_editor.core import roi
import numpy as np

def masks_from_rois(f_roi, im_shape):

    # read rois
    roi_ = roi.ROICollection(im_shape)
    rois = roi_.load_from_json(f_roi)
    rois = rois.rois

    # compute masks from rois
    masks = [rois[i].compute_mask(im_shape).astype(np.uint8)*255 for i in range(len(rois))]

    # compute union of masks
    mask = np.any(masks, axis=0).astype(np.uint8)*255

    return masks, mask
