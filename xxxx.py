import cv2
import numpy as np

img = cv2.imread("MicrosoftTeams-image.png")

dh, dw, _ = img.shape
box = "0 0.459229 0.613525 0.163574 0.0864258"
class_id, x_center, y_center, w, h = box.strip().split()
x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
x_center = round(x_center * dw)
y_center = round(y_center * dh)
w = round(w * dw)
h = round(h * dh)
x = round(x_center - w / 2)
y = round(y_center - h / 2)

imgCrop = img[y:y + h, x:x + w]

cv2.imshow('imga', imgCrop)
cv2.waitKey()