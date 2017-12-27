#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_image
from matplotlib.widgets import Slider, Button, RadioButtons

# 使用一個全域字典來儲存已經計算過得模糊圖形
dict_blurred = {}

def bgr_to_rgb(input_img):
    # OpenCV 的預設影像色彩順序為 BGR, 而 Matplotlib 是 RGB.
    return cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

def to_blur(image, size):
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (size, size), 0)
    return blurred_image

def update_image(axs, ori, blur):
    axs[0].set_title('Original')
    axs[0].imshow(bgr_to_rgb(ori))
    axs[1].set_title('Averaged')
    axs[1].imshow(bgr_to_rgb(blur))

def show_blurs(original):
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
    # https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
    fig, axs = plt.subplots(1, 2)

    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html#matplotlib.pyplot.subplots_adjust
    # 設定 subplot 的 layout
    plt.subplots_adjust(bottom=0.25)

    global dict_blurred
    blurred = dict_blurred.get(7, None)
    if not blurred:
        blurred = to_blur(original, 7)
    update_image(axs, original, blurred)

    # 設定軸色
    axcolor = 'lightgoldenrodyellow'
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes
    # 新增拖曳槓的軸線圖, left/bottom/width/height
    axfs = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

    # https://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Slider
    # 連結對應的軸線, 設定最小值 1, 最大值 19, 初始值 5
    sfs = Slider(axfs, 'Filter Size', 1, 19, valinit=9)
    sfs.valtext.set_text(9)


    def update_size(val):
        final_val = int(val)
        if final_val % 2 == 0:
            final_val += 1
        global dict_blurred
        blur = dict_blurred.get(final_val, None)
        if blur is None:
            blur = to_blur(original, final_val)
            dict_blurred[final_val] = blur
        sfs.valtext.set_text(final_val)
        update_image(axs, original, blur)
    # 設定 Slide 值變動時的 callback 函數
    sfs.on_changed(update_size)

    # 新增重置按鈕的軸線圖
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # 連結按鈕軸線圖與按鈕
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.5')
    def reset_slider(event):
        sfs.reset()
    # 當按鈕被按下之後, 重置 Slider 數值
    button.on_clicked(reset_slider)

    plt.show()
    pass

def thresholding(original):
    # NOTE : 進行 Threshold 計算的影像應該要為灰階影像
    gray_ori = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    # THRESH_BINARY : 超過閥值(第二參數)者設為 maxval(第三參數), 小於閥值者設為 0
    ret, thresh1 = cv2.threshold(gray_ori, 127, 255, cv2.THRESH_BINARY)
    # THRESH_BINARY_INV : 超過閥值(第二參數)者設為 0, 小於閥值者設為 maxval(第三參數)
    ret, thresh2 = cv2.threshold(gray_ori, 127, 255, cv2.THRESH_BINARY_INV)
    # THRESH_TRUNC : 超過閥值(第二參數)者設為 閥值, 小於閥值者設為 原值
    ret, thresh3 = cv2.threshold(gray_ori, 127, 255, cv2.THRESH_TRUNC)
    # THRESH_TOZERO : 超過閥值(第二參數)者設為 原值, 小於閥值者設為 0
    ret, thresh4 = cv2.threshold(gray_ori, 127, 255, cv2.THRESH_TOZERO)
    # THRESH_TOZERO_INV : 超過閥值(第二參數)者設為 0, 小於閥值者設為 原值
    ret, thresh5 = cv2.threshold(gray_ori, 127, 255, cv2.THRESH_TOZERO_INV)

    thresh6 = cv2.adaptiveThreshold(gray_ori, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                    cv2.THRESH_BINARY, 13, 5)
    thresh7 = cv2.adaptiveThreshold(gray_ori, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY, 13, 5)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV',
              'AdaptiveMeanC', 'AdaptiveGaussianC']
    images = [gray_ori, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]

    for i in range(8):
        plt.subplot(3,3,i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB))
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

def run(image_path):
    # png => rgb
    # jpg => bgr
    img_data = cv2.imread(image_path)
    show_blurs(img_data)
    thresholding(img_data)