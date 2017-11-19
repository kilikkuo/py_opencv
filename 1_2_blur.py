#coding=utf-8

import cv2
from utils import show_image

def to_blur(image):
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (7,7), 0)
    return blurred_image


def run():
    img = 'clustering.png'
    img_data = cv2.imread(img)

    blurred_img = to_blur(img_data)

    show_image(blurred_img)