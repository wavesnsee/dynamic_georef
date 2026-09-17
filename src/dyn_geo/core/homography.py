import numpy as np
from pathlib import Path
import json
import cv2
from shapely import Polygon

from dyn_geo.core.feature_matching import read_matches, save_matches
from dyn_geo.core.mask import pts_inside
from roi_editor.core import roi


def save_h(H, outdir, stem):
    name = stem + '.npy'
    np.save(outdir / name, H)
    return


def run(path, f_roi_low_distort: Path):

    # read raw matches
    ls, df_m = read_matches(path.matches_data_raw)

    # read roi of low distorsion
    with open(f_roi_low_distort, 'r') as f:
        r = json.load(f)
    img_shape = r['img_shape']
    roi_ = roi.ROICollection(img_shape)
    roi_ld = roi_.load_from_json(f_roi_low_distort)
    roi_ld = Polygon(roi_ld.rois[0].points)

    # loop through matching pairs
    for i, df in enumerate(df_m):

        # Keep src, dst points inside roi of low distorsion
        mask_src = pts_inside(roi_ld, df, x='src_x', y= 'src_y')
        mask_dst = pts_inside(roi_ld, df, x='dst_x', y= 'dst_y')
        mask = np.logical_and(mask_src, mask_dst)
        df = df[mask]
        src_pts = df[['src_x', 'src_y']].to_numpy()
        dst_pts = df[['dst_x', 'dst_y']].to_numpy()

        # reshape src and dst pts to (n, 1, 2)
        src_pts = np.reshape(src_pts, (len(src_pts), 1, 2))
        dst_pts = np.reshape(dst_pts, (len(dst_pts), 1, 2))

        # compute homography and inliers
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

        # save homography
        save_h(H, path.h, ls[i].stem)

        # save filtered matching points
        save_matches(src_pts, dst_pts, path.matches_data_filtered, ls[i].stem, valid_mask=np.squeeze(inlier_mask))

    return



