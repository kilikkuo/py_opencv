#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import show_image, set_plt_autolayout
from matplotlib.widgets import Slider, RadioButtons

def orb_brute_force_match():
    img1 = cv2.imread('./HLG.png',0)       # queryImage
    img2 = cv2.imread('./HLG_R.png',0)     # trainImage

    # 建立 ORB 偵測器
    orb = cv2.ORB_create()
    # 透過 ORB 找尋 key points / descs
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 建立 Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 配對 desc1 與 desc2
    matches = bf.match(des1, des2)
    # 將配對結果依距離做排序
    matches = sorted(matches, key=lambda x: x.distance)
    # 畫出前 10 個配對的結果
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    plt.imshow(img3)
    plt.show()
    pass

def orb_descriptor(ori_img):
    to_show = ori_img.copy()
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    # 建立 ORB 偵測器
    orb = cv2.ORB_create()
    # 透過 ORB 找尋 key points
    kps = orb.detect(gray, None)
    # 計算這些 key points 的 descriptions
    kps_new, des = orb.compute(gray, kps)
    print(orb.descriptorSize())
    print(des.shape)
    print("Total Keypoints after ORB: {}".format(len(kps_new)))
    for kp in kps_new:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(to_show, (x, y), 5, (255, 0, 0), -1)

    # GUI 設定
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title('ORB')
    ax.imshow(to_show)
    plt.show()

def brief_descriptor(ori_img):
    to_show = ori_img.copy()
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    # 建立 fast feature 偵測器 (CenSurE)
    star = cv2.xfeatures2d.StarDetector_create()

    # 建立 BRIEF Descriptor Extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # 透過 star 特徵偵測器尋找 key points
    kps = star.detect(gray, None)
    # 計算這些關鍵點的描述
    kps_new, des = brief.compute(gray, kps)
    print(brief.descriptorSize())
    print(des.shape)
    print("Total Keypoints after BRIEF: {}".format(len(kps_new)))
    for kp in kps_new:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(to_show, (x, y), 5, (255, 0, 0), -1)

    # GUI 設定
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title('BRIEF')
    ax.imshow(to_show)
    plt.show()

def detect_fast_corner(ori_img):
    to_show = ori_img.copy()
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    # 建立 fast feature 偵測器
    fast = cv2.FastFeatureDetector_create()
    # 透過 fact 特徵偵測器尋找 key points
    kps = fast.detect(gray, None)

    # 另外一種畫 Keypoint 的作法
    # img2 = cv2.drawKeypoints(ori_img, kps, None, color=(255, 0, 0))
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html
        cv2.circle(to_show, (x, y), 5, (255, 0, 0), -1)

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kps)))

    # GUI 設定
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title('FAST')
    ax.imshow(to_show)
    plt.show()

def detect_good_feature_corner(ori_img):
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = np.float32(gray)

    # GUI 設定
    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title('Result')

    axcolor = 'lightblue'
    axmaxcorners = plt.axes([0.2, 0.20, 0.6, 0.03], facecolor=axcolor)
    axqualitylv = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    axmindistance = plt.axes([0.2, 0.10, 0.6, 0.03], facecolor=axcolor)
    axradio = plt.axes([0.01, 0.5, 0.2, 0.15], facecolor=axcolor)

    rbradio = RadioButtons(axradio, ('Shi-Tomasi', 'Harris'), active=0)

    smc = Slider(axmaxcorners, 'Max Corners', 1, 100, valinit=50)
    sql = Slider(axqualitylv, 'Quality level', 0.01, 0.1, valinit=0.01)
    smd = Slider(axmindistance, 'Min Distance', 1, 20, valinit=10)

    def update(val):
        res_img = ori_img.copy()
        mc = int(smc.val)
        md = int(smd.val)
        ql = sql.val
        useHarris = False if rbradio.value_selected == 'Shi-Tomasi' else True

        smc.valtext.set_text(mc)
        sql.valtext.set_text(ql)
        smd.valtext.set_text(md)

        # 利用 goodFeaturesToTrack 直接尋找需要特徵數量
        corners = cv2.goodFeaturesToTrack(gray, mc, ql, md, useHarrisDetector=useHarris)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(res_img, (x, y), 5, 255, -1)
        ax.imshow(res_img)
        fig.canvas.draw_idle()

    # 設定 Slide 值變動時的 callback 函數
    smc.on_changed(update)
    sql.on_changed(update)
    smd.on_changed(update)

    def change_algo(label):
        update(None)

    rbradio.on_clicked(change_algo)

    update(None)
    plt.show()

def run(img_path):
    img = cv2.imread(img_path)

    detect_good_feature_corner(img)
    detect_fast_corner(img)
    brief_descriptor(img)
    orb_descriptor(img)
    orb_brute_force_match()
    pass
