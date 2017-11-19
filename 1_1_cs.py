#coding=utf-8

import cv2
import numpy
from utils import show_image

def to_gray(image):
    # Convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def to_hsv(image):
    # Convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def extract_green_in_hsv(image):
    # 在 BGR 色彩空間中定義綠色
    bgr_green = numpy.uint8([[[0, 255, 0]]])
    # 將 BGR 色彩空間中的綠色轉換至 HSV 色彩空間
    hsv_green = cv2.cvtColor(bgr_green, cv2.COLOR_BGR2HSV)[0][0]

    # 建立兩個不同的綠色(HSV)
    lower_green = numpy.uint8([hsv_green[0] - 20, 0, 0])
    upper_green = numpy.uint8([hsv_green[0] + 20, 255, 255])

    # 製造一個遮罩, 過濾的值介於上述兩個綠色之間
    mask = cv2.inRange(image, lower_green, upper_green)
    show_image(mask, 'mask', False)

    # 執行 bitwise operation, 將遮罩與圖交集.
    result = cv2.bitwise_and(image, image, mask= mask)
    show_image(result, title='result')
    pass

def run():
    img = 'clustering.png'
    img_data = cv2.imread(img)
    test = cv2.UMat(img_data)
    print(test)
    # show_image(img_data, 'ori', False)

    # gray_img = to_gray(img_data)
    # show_image(gray_img)

    # hsv_img = to_hsv(img_data)
    # show_image(hsv_img, 'hsv', False)
    # extract_green_in_hsv(hsv_img)