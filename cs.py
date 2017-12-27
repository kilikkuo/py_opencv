#coding=utf-8

import cv2
import numpy
from utils import show_image

# OpenCV 支援的色彩空間 enumeration
# https://docs.opencv.org/3.3.0/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0

def to_gray(image):
    # Convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def to_hsv(image):
    # Convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def bgr_to_xyz_rec709_d65(image):
    # Convert to xyz
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    return xyz_image

def rec709_d65_xyz_to_bgr(image):
    # Convert to bgr
    bgr_image = cv2.cvtColor(image, cv2.COLOR_XYZ2BGR)
    return bgr_image

def extract_green_in_hsv(image):
    # 在 BGR 色彩空間中定義綠色
    bgr_green = numpy.uint8([[[0, 255, 0]]])
    # 將 BGR 色彩空間中的綠色轉換至 HSV 色彩空間
    hsv_green = cv2.cvtColor(bgr_green, cv2.COLOR_BGR2HSV)
    hue_green = hsv_green[0][0][0]

    # 建立兩個不同的綠色(HSV)
    lower_green = numpy.uint8([hue_green - 10, 100, 100])
    upper_green = numpy.uint8([hue_green + 10, 255, 255])

    # 製造一個遮罩, 過濾的值介於上述兩個綠色之間
    mask = cv2.inRange(image, lower_green, upper_green)
    show_image(mask, 'Mask', False)

    # 執行 bitwise operation, 將遮罩與圖交集.
    result = cv2.bitwise_and(image, image, mask= mask)
    rgb_result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    show_image(rgb_result, title='Result')
    pass

def run(sample_img):
    img_data = cv2.imread(sample_img)
    show_image(img_data, 'Original', False)

    gray_img = to_gray(img_data)
    show_image(gray_img, 'Gray')

    hsv_img = to_hsv(img_data)
    show_image(hsv_img, 'HSV')

    xyz_img = bgr_to_xyz_rec709_d65(img_data)
    show_image(xyz_img, 'BGR to XYZ')
    bgr_img = rec709_d65_xyz_to_bgr(xyz_img)
    show_image(bgr_img, 'XYZ to BGR')

    extract_green_in_hsv(hsv_img)
