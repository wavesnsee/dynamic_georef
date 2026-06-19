import cv2
import matplotlib.pyplot as plt


def read(f):
    im = cv2.imread(f)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w, h = im.shape[0:2]
    return im, im_gray, w, h
