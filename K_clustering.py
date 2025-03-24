import cv2
import numpy as np

def K_clustering(image,K=2):
    # Applatir l'image pour K-means (transformation en tableau 1D)
    pixels = image.reshape((-1,1)).astype(np.float32)

    # Définir les critères d'arrêt du K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Application du K-means
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruction de l'image segmentée
    segmented = labels.reshape(image.shape)

    # Affichage de l'image segmentée (binaire)
    segmented_display = (segmented * 255).astype(np.uint8)
    return segmented_display