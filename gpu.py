#coding=utf-8

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

get_counter = time.time if sys.version_info[0] == 2 else time.perf_counter

def cpu_version(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 套用高斯模糊降噪 (opt)
    # temp = cv2.GaussianBlur(temp, (11, 11), 0.1)

    # 執行 Canny 演算法
    edge = cv2.Canny(temp, 100, 200, L2gradient=True)
    w, h = edge.shape[1], edge.shape[0]

    cv2.namedWindow("CPU", 0)
    cv2.moveWindow("CPU", 0, 0)
    cv2.resizeWindow("CPU", int(w/4), int(h/4))
    cv2.imshow("CPU", edge)

def gpu_version(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 套用高斯模糊降噪 (opt)
    # temp = cv2.GaussianBlur(temp, (11, 11), 0.1)

    # 執行 Canny 演算法
    uedge = cv2.Canny(temp, 100, 200, L2gradient=True)
    edge = uedge.get()
    w, h = edge.shape[1], edge.shape[0]

    cv2.namedWindow("GPU", 0)
    cv2.moveWindow("GPU", 0, 0)
    cv2.resizeWindow("GPU", int(w/4), int(h/4))
    cv2.imshow("GPU", edge)

def run(image_path):
    # Choose device from either CPU or GPU. 不設定的話 OpenCV 不會真的啟動 opencl.
    os.environ['OPENCV_OPENCL_DEVICE'] = ':CPU|GPU:'

    print(' Does platform support OpenCL : {}'.format(cv2.ocl.haveOpenCL()))
    print(' setUseOpenCL(True)')
    cv2.ocl.setUseOpenCL(True)
    print(' Can OpenCV use OpenCL : {}'.format(cv2.ocl.useOpenCL()))

    img_mat = cv2.imread(image_path, 1)
    # 將 image data 轉換成 UMat 形式, upload data to memory where GPU can access
    img_umat = cv2.UMat(img_mat)

    # 利用 perf_count (CPU 的 counter, 依據 CPU 的硬體時鐘以一定的頻率單向增加, counter
    # 的值在CPU power-off 或 reset 時會被重置), 屬於相對時間, 量測程式碼較準確, 因為其數值
    # 的精細度來自於執行指令數目.
    # time.time() 使用的是 dedicated hardware 時鐘, 通常是使用電池支撐的. 屬於絕對時間
    s1 = get_counter()
    cpu_version(img_mat)
    s2 = get_counter()
    print(' cpu : {}'.format(s2-s1))

    s3 = get_counter()
    gpu_version(img_umat)
    s4 = get_counter()
    print(' gpu : {}'.format(s4-s3))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
