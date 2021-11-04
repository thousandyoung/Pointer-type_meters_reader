import os
import cv2
import numpy as np
#--- performs Otsu threshold ---
def threshold(img, st):
    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    # ret, thresh = cv2.threshold(img, 0, 255,  cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    return  img

