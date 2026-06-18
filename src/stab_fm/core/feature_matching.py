from roi_editor.core import roi
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def recover_matching_keypoints(
        ref_fn,
        ref_f_rois,
        target_imgs_dir,
        outdir):

    # read reference image
    im_ref =  cv2.imread(ref_fn)
    # im_ref = cv2.cvtColor(im_ref, cv2.COLOR_BGR2RGB)
    im_ref_gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)
    rows, cols, _ = im_ref.shape

    # get masks from rois that were defined on ref image
    masks, mask_ref = masks_from_rois(ref_f_rois, im_ref.shape[0:2])

    # compute keypoints and descriptors of ref img
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kps_ref = []
    des_ref = []
    for i in range(len(masks)):
        kp, des = sift.detectAndCompute(im_ref, masks[i])
        kps_ref.append(kp)
        des_ref.append(des)

    # dilate masks for target images
    sz_dil = 800
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz_dil, sz_dil))
    masks_target = [cv2.dilate(masks[i], kernel, iterations=1) for i in range(len(masks))]

    # outdirs
    outdir_matches = outdir / 'matches'
    outdir_matches.mkdir(parents=True, exist_ok=True)
    outdir_matches_plots = outdir_matches / 'plots'
    outdir_matches_plots.mkdir(parents=True, exist_ok=True)
    outdir_warped = outdir / 'warped'
    outdir_warped.mkdir(parents=True, exist_ok=True)

    # initialize list of homography matrixes
    H = []

    # loop through target images
    ls = sorted(target_imgs_dir.glob('*.jp*g'))
    for f in ls:
        print(f)

        # initialize source and destination points
        dst_pts = []
        src_pts = []

        # read target image
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # loop through masks
        for i in range(len(masks)):
            # compute keypoints and descriptors
            kp, des = sift.detectAndCompute(im, masks_target[i])
            raw_matches = bf.knnMatch(des_ref[i], des, k=2)
            # filter matching keypoints by a distance criteria
            good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
            dst_pts.append(np.float32([kps_ref[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2))
            src_pts.append(np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))

        # combine multiple arrays of shape (n, 1, 2) into a single array of shape (total_n, 1, 2)
        dst_pts = np.vstack(dst_pts)
        src_pts = np.vstack(src_pts)

        # compute homography and inliers
        h, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
        H.append(h)

        # plot source and destination matches
        dst_pts = np.squeeze(dst_pts)
        src_pts = np.squeeze(src_pts)
        inlier_mask = np.squeeze(inlier_mask)
        inlier_inds = np.where(inlier_mask == 1)
        fig, ax = plt.subplots(1,2, figsize=(22, 12), sharex=True, sharey=True, tight_layout=True)
        ax[0].imshow(im_ref)
        ax[1].imshow(im)
        ax[0].plot(dst_pts[:, 0], dst_pts[:, 1], c='g', linewidth=0, markersize=6, marker='s')
        ax[1].plot(src_pts[:, 0], src_pts[:, 1], c='r', linewidth=0, markersize=6, marker='d')
        ax[1].plot(src_pts[inlier_inds, 0], src_pts[inlier_inds, 1], c='b', linewidth=0, markersize=6, marker='d')
        fig.savefig(outdir_matches_plots / f.name)

        # apply homography to target image
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # warped_img = cv2.warpPerspective(im, h, (cols, rows))

        # ECC
        im_ref_gray = im_ref_gray.astype(np.float32)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        warp_mode = cv2.MOTION_HOMOGRAPHY
        # The warp matrix is the initial homography, converted to float32 if needed
        warp_matrix = h.astype(np.float32)
        # Termination criteria: stop after 1000 iterations or when epsilon is reached
        eps = 0.001
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, eps)
        # The order of images: (templateImage, inputImage, warpMatrix, ...)
        # It will refine warp_matrix to best align inputImage to templateImage
        mask = cv2.dilate(mask_ref, kernel, iterations=1)
        # cc, refined_h = cv2.findTransformECC(im_ref_gray, im_gray, warp_matrix, warp_mode, criteria, None, mask)
        try:
            cc, refined_h = cv2.findTransformECCWithMask(im_ref_gray, im_gray, mask_ref, mask, warp_matrix, warp_mode, criteria)
            # apply homography to target image
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            warped_img = cv2.warpPerspective(im, refined_h, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
            # save warped image
            cv2.imwrite(outdir_warped / f.name, warped_img)
        except:
            print('ecc iterations did not converge')

    # return src_pts, dst_pts, H
    return