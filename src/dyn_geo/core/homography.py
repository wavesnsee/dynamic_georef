import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

from dyn_geo.core.feature_matching import read_matches, save_matches
from dyn_geo.core.mask import read_polygon_roi, pts_inside


def save_h(H, outdir, stem):
    name = stem + '.npy'
    np.save(outdir / name, H)
    return


def compute(dir_matches_data_raw: Path, start: datetime, end: datetime, dir_h: Path, dir_matches_data_filtered: Path,
            f_roi_low_distort: Path):

    # read raw matches
    ls, df_m = read_matches(dir_matches_data_raw, start, end)

    # read roi of low distorsion
    roi_ld = read_polygon_roi(f_roi_low_distort)

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
        if len(src_pts) >= 4:
            H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

            # save homography
            save_h(H, dir_h, ls[i].stem)

            # save filtered matching points
            save_matches(src_pts, dst_pts, dir_matches_data_filtered, ls[i].stem, valid_mask=np.squeeze(inlier_mask))

    return


