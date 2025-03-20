import cv2



"""
img=cv2.imread("C:\\Users\\antod\\Desktop\\Projet S6\\ia-detection-leucemie\\exemple_image.jpg")
resized_img = cv2.resize(img, (224, 224))
cv2.imshow('image initial', img)
cv2.imshow('image de taille 224x224', resized_img)
cv2.waitKey(0)
"""


def resizing1(image_jpg):
    """fonction qui permet de convertir la taille de l'image rentrée en paramètre au format 224 x 224
    Arguments :
    image_jpg : nom de l'image que l'on veut convertir (se termine par .jpg)
    """


    img=cv2.imread(f"C:\\Users\\antod\\Desktop\\Projet S6\\ia-detection-leucemie\\{image_jpg}")
    resized_img = cv2.resize(img, (224, 224))
    cv2.imshow('image initial', img)
    cv2.imshow('image de taille 224x224', resized_img)
    cv2.waitKey(0)
    return


resizing1("exemple_image.jpg")