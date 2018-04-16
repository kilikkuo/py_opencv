#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_image, set_plt_autolayout
from matplotlib.widgets import Slider, Button

def detect_apple(ori_img):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    axcolor = 'lightblue'
    # 建立兩個 Axes 給對應的 Slider 使用
    axscalefactor = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    axminneighbor = plt.axes([0.2, 0.10, 0.6, 0.03], facecolor=axcolor)
    # 建立兩個 Slider 
    ssf = Slider(axscalefactor, 'Scale Factor', 1, 1.1, valinit=1.05)
    smn = Slider(axminneighbor, 'Min Neighbor', 2, 10, valinit=4)

    # 下載預先訓練好的分類器
    apple_cascade = cv2.CascadeClassifier('./data/totrain/traincascade/cascade.xml')
    # 將影像轉灰階
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    def update(val):
        to_show = ori_img.copy()
        sf = ssf.val
        mn = int(smn.val)
        # 開始進行蘋果偵測
        apples = apple_cascade.detectMultiScale(gray, sf, mn)
        for x, y, w, h in apples:
            cv2.rectangle(to_show, (x, y), (x+w, y+h), (255, 0, 0), 3)

        ax.set_title("Apple")
        ax.imshow(cv2.cvtColor(to_show, cv2.COLOR_BGR2RGB))

    ssf.on_changed(update)
    smn.on_changed(update)
    update(None)

    plt.show()
    pass

def run(image_path):
    img = cv2.imread(image_path)
    detect_apple(img)
    pass
