#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from utils import show_image

def multi_matching(img_path, template_path):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_bgr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    axcolor = 'yellow'
    # 新增拖曳槓的軸線圖, left/bottom/width/height
    axthreshold1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    # 連結對應的軸線, 設定最小值 0, 最大值 1, 初始值 0.5
    sth1 = Slider(axthreshold1, 'Threshold', 0, 1, valinit=0.5)
    def update_found(val):
        threshold = sth1.val
        loc = np.where(res >= threshold)
        img_bgr_2 = img_bgr.copy()
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_bgr_2, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

        rgb_img = cv2.cvtColor(img_bgr_2, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_img)

    sth1.on_changed(update_found)
    update_found(0.5)
    plt.show()

def template_matching(img_path, template_path):
    ori_img = cv2.imread(img_path)
    # 原圖轉灰階
    ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    # 透過 img.shape 可以得到影像的 Shape. 返回值為 tuple (row, columns, channels)
    # [::-1] ==> 將序列反向
    w, h = template.shape[::-1]

    # 將所有提供的比較法一一列出
    # https://docs.opencv.org/3.3.0/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
    dic_methods = {cv2.TM_CCOEFF : 'cv2.TM_CCOEFF',
                   cv2.TM_CCOEFF_NORMED : 'cv2.TM_CCOEFF',
                   cv2.TM_CCORR : 'cv2.TM_CCORR',
                   cv2.TM_CCORR_NORMED : 'cv2.TM_CCORR_NORMED',
                   cv2.TM_SQDIFF : 'cv2.TM_SQDIFF',
                   cv2.TM_SQDIFF_NORMED : 'cv2.TM_SQDIFF_NORMED',}

    for enum_m, str_m in dic_methods.items():
        # 套用樣板比對方式
        res = cv2.matchTemplate(ori_gray, template, enum_m)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # 若使用 TM_SQDIFF or TM_SQDIFF_NORMED, 取最小值為最近似區
        if enum_m in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # 將比對最接近區域劃上白色 (255,255,255)/BGR 方框, 並且指定框框的厚度 2
        cv2.rectangle(ori_img, top_left, bottom_right, (255, 255, 255), 2)

        plt.subplot(121)
        plt.imshow(res, cmap='gray')
        plt.title('Matching')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(str_m)

        plt.show()

def run(img_path, template_path):
    # template_matching(img_path, template_path)
    multi_matching(img_path, template_path)
    pass