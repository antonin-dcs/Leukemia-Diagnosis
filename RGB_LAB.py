import cv2
import numpy as np

# Charger l'image
image = cv2.imread("Exemple_image.jpg")

# Convertir l'image de RGB à LAB
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# Extraire le canal A (qui met en évidence les cellules B-ALL)
lab_a = lab_image[:,:,1]

# Afficher l'image originale, la conversion LAB et le canal A
cv2.imshow("RGB", image)
cv2.imshow("LAB", lab_image)
cv2.imshow("LAB - A", lab_a)

# Applatir l'image pour K-means (transformation en tableau 1D)
pixels = lab_a.reshape((-1,1)).astype(np.float32)

# Définir les critères d'arrêt du K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 4  # On divise en 2 groupes

# Application du K-means
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reconstruction de l'image segmentée
segmented = labels.reshape(lab_a.shape)

# Affichage de l'image segmentée (binaire)
segmented_display = (segmented * 255).astype(np.uint8)
cv2.imshow("Segmented Image (binary)", segmented_display)

# --- Ajout du seuil binaire (binary thresholding) ---
# Définir un seuil binaire
threshold_value = 127  # Seuil à définir selon l'intensité désirée

# Appliquer un seuil binaire sur l'image segmentée
_, binary_image = cv2.threshold(segmented_display, threshold_value, 255, cv2.THRESH_BINARY)

# Afficher l'image binaire résultante
cv2.imshow("Binary Thresholded Image", binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
