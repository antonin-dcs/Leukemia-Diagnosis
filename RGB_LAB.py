import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:\\Users\\Utilisateur\\Desktop\\2024-2025\\Projet S6\\Exemple_image.jpg")

# Convertir l'image de RGB à LAB
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# Extraire le canal A (qui met en évidence les cellules B-ALL)
lab_a = lab_image[:,:,1]  

# Afficher l'image LAB-A
cv2.imshow("LAB - A", lab_a)
cv2.waitKey(0)
cv2.destroyAllWindows()