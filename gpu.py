#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from utils import show_image
import time


def plot_result(ori, edged, useGPU=False):
    typeOfDevice = 'GPU' if useGPU else 'CPU'
    fig, list_ax = plt.subplots(1, 2)
    list_ax[0].set_title('{} Original Image'.format(typeOfDevice))
    # 使用 matplotlib 畫灰階圖時必須設定 color map 才能得到正確結果
    list_ax[1].set_title('{} Edged Image'.format(typeOfDevice))

    ori_img = ori.get() if useGPU else ori
    edge_img = edged.get() if useGPU else edged

    list_ax[0].imshow(ori_img, cmap='gray')
    list_ax[1].imshow(edge_img, cmap='gray')

def cpu_version(img):
    # 套用高斯模糊降噪 (opt)
    ori_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 執行 Canny 演算法
    edge = cv2.Canny(ori_blurred, 100, 200, L2gradient=True)
    plot_result(ori_blurred, edge, False)

def gpu_version(img):
    # 套用高斯模糊降噪 (opt)
    ori_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 執行 Canny 演算法
    edge = cv2.Canny(ori_blurred, 100, 200, L2gradient=True)

    plot_result(ori_blurred, edge, True)

def run(image_path):
    img_mat = cv2.imread(image_path, 0)
    # 將 image data 轉換成 UMat 形式, upload data to memory where GPU can access
    img_umat = cv2.UMat(img_mat)

    # 利用 perf_count (CPU 的 counter, 依據 CPU 的硬體時鐘以一定的頻率單向增加, counter
    # 的值在CPU power-off 或 reset 時會被重置), 屬於相對時間, 量測程式碼較準確, 因為其數值
    # 的精細度來自於執行指令數目.
    # time.time() 使用的是 dedicated hardware 時鐘, 通常是使用電池支撐的. 屬於絕對時間
    s1 = time.perf_counter()
    cpu_version(img_mat)
    s2 = time.perf_counter()
    gpu_version(img_umat)
    s3 = time.perf_counter()
    print(' cpu : {}'.format(s2-s1))
    print(' gpu : {}'.format(s3-s2))
    plt.show()
