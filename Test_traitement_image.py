import numpy as np
import cv2
from resising_224x224 import *
from RGB_LAB import *
from K_clustering import *
from binary_thresholding import *
from Retour_RGB import *

image_RGB=resizing1("exemple_image.jpg")
image_traitee=Binary_tresholding(K_clustering(RGB_LAB(resizing1("exemple_image.jpg"))))
cv2.imshow("image de base",image_RGB)
cv2.imshow('Image traitée', image_traitee)
cv2.imshow('Retour aux origines',mask_RGB(image_RGB,image_traitee))
cv2.waitKey(0)
cv2.destroyAllWindows()