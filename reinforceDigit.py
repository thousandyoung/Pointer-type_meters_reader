import os
import cv2
import numpy as np

#--- performs Otsu threshold ---
def threshold(img, st):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return  thresh


def reinforceDigit(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)   #--- resized the image because it was to big
    cv2.imshow('Original', img)

    #--- see each of the channels individually ---
    cv2.imshow('b', img[:,:,0])
    cv2.imshow('g', img[:,:,1])
    cv2.imshow('r', img[:,:,2])

    m1 = threshold(img[:,:,0], 1)   #--- threshold on blue channel
    m2 = threshold(img[:,:,1], 2)   #--- threshold on green channel
    m3 = threshold(img[:,:,2], 3)   #--- threshold on red channel

    #--- adding up all the results above ---
    res = cv2.add(m1, cv2.add(m2, m3))

    cv2.imshow('res', res)
    cv2.imwrite('images/res.jpg', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 'images/res.jpg'