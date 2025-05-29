import cv2 as cv

def remove_isolated_pixels(img,k=7):
    """k est la taile du motif pour supprimer le bruit avec la focntion otsu"""
    blur = cv.GaussianBlur(img,(k,k),0)
    ret3,img_th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return img_th3


    