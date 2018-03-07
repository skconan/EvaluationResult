'''
    File name: lib.py
    Author: skconan
    Date created: 2018/03/06
    Date last modified: 2018/03/07
    Python Version: 3.6.1
'''

import cv2
import numpy as np
import operator


def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    elif shape == '\\':
        kernel = np.diag([1] * ksize[0])
        return np.uint8(kernel)
    elif shape == '/':
        kernel = np.fliplr(np.diag([1] * ksize[0]))
        return np.uint8(kernel)
    else:
        return None


def clahe_gray(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    resGRAY = clahe.apply(img_gray)
    return resGRAY
