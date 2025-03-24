import cv2
import numpy as np

def Binary_tresholding(image, seuil=127):
    # Appliquer un seuil binaire sur l'image segmentée
    _, binary_image = cv2.threshold(image, seuil, 255, cv2.THRESH_BINARY)
    return binary_image
