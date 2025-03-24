
import numpy as np
import cv2

def remove_isolated_pixels(src):
    kernel = np.ones((3, 3), np.uint8)  # Kernel 3x3 pour compter les voisins
    neighbor_count = cv2.filter2D(src, ddepth=-1, kernel=kernel)  # Comptage des pixels blancs autour de chaque pixel
    
    # Masques pour les pixels isolés
    isolated_white = (src == 255) & (neighbor_count <= 255 * 2)  # Pixels blancs isolés
    isolated_black = (src == 0) & (neighbor_count >= 255 * 7)    # Pixels noirs isolés
    
    # Création d'une copie et correction des pixels isolés
    result = src.copy()
    result[isolated_white] = 0  # Remplace les pixels blancs isolés par du noir
    result[isolated_black] = 255  # Remplace les pixels noirs isolés par du blanc
    
    return result
