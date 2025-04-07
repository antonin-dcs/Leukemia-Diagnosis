import cv2 as cv
import numpy as np

def Binary_tresholding(image, seuil=127):
    # Appliquer un seuil binaire sur l'image segmentée
    _, binary_image = cv.threshold(image, seuil, 255, cv.THRESH_BINARY)
    return image

def binary_tresholding_mean(img):
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                               cv.THRESH_BINARY,11,2)
    return th2 