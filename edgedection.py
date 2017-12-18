#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from utils import show_image

import platform
current_os = platform.system()

# Canny 演算法基於對單通道灰階影像而設計, 將輸入以灰階圖方式讀入
ori = cv2.imread('giant.jpg', 0)
# 套用高斯模糊降噪 (opt)
ori_blurred = cv2.GaussianBlur(ori, (5,5), 0)
# 執行 Canny 演算法
edge = cv2.Canny(ori_blurred, 100, 200, L2gradient=True)

def edge_detection_only(o_img, e_img):
    show_image(o_img, 'Original Image', False)
    show_image(e_img, 'Edged Image')

def edge_detection_with_controls(o_img, e_img):
    # 建立 1 row, 2 column 的 subplot
    fig, list_ax = plt.subplots(1,2)

    # 設定 subplot 的 layout
    plt.subplots_adjust(left=0.25, bottom=0.25)
    list_ax[0].set_title('Original Image')
    # 使用 matplotlib 畫灰階圖時必須設定 color map 才能得到正確結果
    list_ax[0].imshow(o_img, cmap='gray')
    list_ax[1].set_title('Edged Image')
    list_ax[1].imshow(e_img, cmap='gray')

    # 設定軸色
    axcolor = 'yellow'
    # 新增拖曳槓的軸線圖, left/bottom/width/height
    axthreshold1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axthreshold2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

    # 連結對應的軸線, 設定最小值 1, 最大值 100, 初始值 50
    sth1 = Slider(axthreshold1, 'Threshold 1', 1, 200, valinit=100)
    # 連結對應的軸線, 設定最小值 50, 最大值 250, 初始值 150
    sth2 = Slider(axthreshold2, 'Threshold 2', 100, 500, valinit=200)
    def update(val):
        list_ax[1].clear()
        e_img = cv2.Canny(o_img, sth1.val, sth2.val)
        list_ax[1].set_title('Edged Image')
        list_ax[1].imshow(e_img, cmap='gray')

    # 設定 Slide 值變動時的 callback 函數
    sth1.on_changed(update)
    sth2.on_changed(update)

    # 新增重置按鈕的軸線圖
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # 連結按鈕軸線圖與按鈕
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.5')
    def reset_slider(event):
        sth1.reset()
        sth2.reset()
    # 當按鈕被按下之後, 重置 Slider 數值
    button.on_clicked(reset_slider)

    plt.show()

def get_keypress():
    print('Enter 1 : 執行邊緣偵測')
    print('Enter 2 : 執行邊緣偵測(加上控制項)')
    key = ''
    if current_os == 'Linux':
        raw_key = input()
        key = raw_key[0] if len(raw_key) >= 1 else chr(27)
        ord_key = ord(key)
        # '1'
        if ord_key == 49:
            return True, edge_detection_only
        # '2'
        elif ord_key == 50:
            return True, edge_detection_with_controls
        # 'esc'
        elif ord_key == 27:
            return False, None

    return False, None

def run():
    result, func = get_keypress()
    if result:
        func(ori_blurred, edge)
