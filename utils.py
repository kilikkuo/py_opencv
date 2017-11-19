#coding=utf-8

import cv2

def show_image(image, title='image', wait=True):
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
