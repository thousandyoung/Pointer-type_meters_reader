import os
import cv2
import numpy as np
#--- performs Otsu threshold ---
def threshold(img, st):
    ret, thresh = cv2.threshold(img, 0, 255,  cv2.THRESH_OTSU)
    return  thresh

