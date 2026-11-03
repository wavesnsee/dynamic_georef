import cv2
import numpy as np

from dyn_geo.core.img import ls_period


def run(target_imgs_dir, start, end, dir_h, dir_warped):

    # list of npy homography matrixes
    ls = ls_period(dir_h, start, end,  extension='*.npy')

    for f_h in ls:

        # load homography matrix
        H = np.load(f_h)

        # read target im
        im = cv2.imread(target_imgs_dir / (f_h.stem + '.jpg'))
        h, w = im.shape[0:2]

        # warp im
        warped_im = cv2.warpPerspective(im, H, (w, h))

        # save warped image
        cv2.imwrite(dir_warped / (f_h.stem + '.jpg'), warped_im)

