#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_image, set_plt_autolayout
from matplotlib.widgets import Slider, Button

def detect_face(ori_img):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    axcolor = 'lightblue'
    axscalefactor = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    axminneighbor = plt.axes([0.2, 0.10, 0.6, 0.03], facecolor=axcolor)

    ssf = Slider(axscalefactor, 'Scale Factor', 1, 2, valinit=1.1)
    smn = Slider(axminneighbor, 'Min Neighbor', 1, 10, valinit=3)

    # 下載預先訓練好的分類器
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    def update(val):
        to_show = ori_img.copy()
        sf = ssf.val
        mn = int(smn.val)
        faces = face_cascade.detectMultiScale(gray, sf, mn)
        for x, y, w, h in faces:
            cv2.rectangle(to_show, (x, y), (x+w, y+h), (255, 255, 255), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = to_show[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        ax.set_title("Faces")
        ax.imshow(cv2.cvtColor(to_show, cv2.COLOR_BGR2RGB))

    ssf.on_changed(update)
    smn.on_changed(update)
    update(None)

    plt.show()
    pass

def run(image_path):
    img = cv2.imread(image_path)
    detect_face(img)
    pass
