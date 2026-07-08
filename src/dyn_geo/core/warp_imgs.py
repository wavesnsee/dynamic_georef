import cv2
import numpy as np


def run(target_imgs_dir, dir_h, dir_warped):

    # list of npy homography matrixes
    ls = sorted(dir_h.glob('*.npy'))

    for f_h in ls:
        print(f_h)

        # load homography matrix
        H = np.load(f_h)

        # read target im
        im = cv2.imread(target_imgs_dir / (f_h.stem + '.jpg'))
        h, w = im.shape[0:2]

        # warp im
        warped_im = cv2.warpPerspective(im, H, (w, h))

        # save warped image
        cv2.imwrite(dir_warped / (f_h.stem + '.jpg'), warped_im)

