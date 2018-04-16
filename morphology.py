#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_image, set_plt_autolayout
from matplotlib.widgets import Slider

def do_erosion(img, kernel_size, iterations=1):
    # 侵蝕
    # 如果 kernel_size 為 0, 直接回傳原圖
    if kernel_size == 0:
        return img
    # 建立一個 size X size 且所有 element 為 1 的 kernel 矩陣.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(img, kernel, iterations)
    return erosion

def do_dilation(img, kernel_size, iterations=5):
    # 擴張
    if kernel_size == 0:
        return img
    # 建立一個 size X size 且所有 element 為 1 的 kernel 矩陣.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations)
    return dilation

def do_opening(img, kernel_size):
    # 斷開
    if kernel_size == 0:
        return img
    # opening : 是先套一個擴張(Dilation), 接著再套一個侵蝕(Erosion)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def do_closing(img, kernel_size):
    # 閉合
    if kernel_size == 0:
        return img
    # closing : 是先套一個侵蝕(Erosion), 接著再套一個擴張(Dilation)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def demo_four_operations():
    ori_img = cv2.imread('./morphology_j.png')
    ori_open = cv2.imread('./morphology_opening.png')
    ori_close = cv2.imread('./morphology_closing.png')

    # 根據新的 kernel 矩陣大小計算新的圖, 並畫在 subplot
    def update_subplot(kernel_size):
        erosion = do_erosion(ori_img, kernel_size)
        dilation = do_dilation(ori_img, kernel_size)
        opening = do_opening(ori_open, kernel_size)
        closing = do_closing(ori_close, kernel_size)
        titles = ['Original', '', 'Erosion', 'Dilation', 'Opening', 'Closing']
        imgs = [ori_img, None, erosion, dilation, opening, closing]
        for idx, data in enumerate(imgs):
            if data is not None:
                plt.subplot(3, 2, idx+1)
                plt.imshow(imgs[idx])
                plt.title(titles[idx])
                plt.xticks([])
                plt.yticks([])

    #  ==========================================================
    # 設定 Slider 來控制 morphology 的 kernel 矩陣尺寸, 並重新計算結果
    axcolor = 'lightgoldenrodyellow'
    axks = plt.axes([0.55, 0.80, 0.3, 0.03], facecolor=axcolor)
    init_kernal_size = 0
    sks = Slider(axks, 'Kernel Size', 0, 10, valinit=init_kernal_size)
    sks.valtext.set_text(init_kernal_size)
    def update_kernel_size(val):
        final_val = int(val)
        sks.valtext.set_text(final_val)
        update_subplot(final_val)
    sks.on_changed(update_kernel_size)
    #  ==========================================================

    update_subplot(init_kernal_size)
    plt.show()

def demo_colorblind():
    colorblind_img = cv2.imread('./colorblind.jpg')
    # 繪出原圖
    show_image(colorblind_img)
    # 轉成 HSV
    hsv_img = cv2.cvtColor(colorblind_img, cv2.COLOR_BGR2HSV)
    show_image(hsv_img)

    # 將 HSV 的圖做五次擴張與侵蝕
    count = 0
    while count < 5:
        hsv_img = do_dilation(hsv_img, 3)
        hsv_img = do_erosion(hsv_img, 1)
        show_image(hsv_img, millisec=400)
        count += 1

    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    show_image(hsv_img)

def run():
    demo_four_operations()
    demo_colorblind()
