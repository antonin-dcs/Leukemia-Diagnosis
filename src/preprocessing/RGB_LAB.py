import cv2
import numpy as np

def RGB_LAB(image):
    # Convertir l'image de RGB à LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Extraire le canal A (qui met en évidence les cellules B-ALL)
    lab_a = lab_image[:,:,1]
    return lab_a