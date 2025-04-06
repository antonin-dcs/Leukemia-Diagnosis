import numpy as np
import cv2 

def mask_RGB(image_RGB, image_traitee):
    # Crée le masque en vérifiant que les pixels de image_traitee ne sont pas égaux à 0
    mask = image_traitee!=0
    # Crée une image filtrée avec la même forme que image_RGB
    image_filtered = np.zeros_like(image_RGB)
    # Applique le masque sur l'image RGB
    image_filtered[mask] = image_RGB[mask]
    return image_filtered