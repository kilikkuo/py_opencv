#coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def hist_draw_curve(img, normalized=False):
    h_result = np.zeros((300, 256, 3))

    if len(img.shape) == 2:
        # 灰階影像
        colors = [(255, 255, 255)]
    elif img.shape[2] == 3:
        # 彩色影像
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    else:
        assert False, "Incorrect image shape !"

    for channel, color in enumerate(colors):
        hist_item = cv2.calcHist([img], [channel], None, [256], [0, 256])
        if normalized:
            # 將得到的 historgram 值正規化在 0 ~ 255 之間
            cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)

        # 將給予的 array-like object 內所有 item 都做四捨五入(round)
        hist = np.int32(np.around(hist_item))
        """
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.column_stack.html
        >>> a = np.array((1,2,3))
        >>> b = np.array((2,3,4))
        >>> np.column_stack((a,b))
        array([[1, 2],
               [2, 3],
               [3, 4]])
        """
        xbins = np.arange(256).reshape(256,1)
        pts = np.int32(np.column_stack((xbins, hist)))

        # https://docs.opencv.org/trunk/d6/d6e/group__imgproc__draw.html#ga444cb8a2666320f47f09d5af08d91ffb
        cv2.polylines(h_result, [pts], False, color)

    # 上下方向地翻轉 array 的 item
    return np.flipud(h_result)

def hist_draw_lines(img):
    assert len(img.shape) == 2, "Must be grayscale image."

    h_result = np.zeros((300, 256))
    hist_item = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 將得到的 historgram 值正規化在 0 ~ 255 之間
    cv2.normalize(hist_item, hist_item, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # 將給予的 array-like object 內所有 item 都做四捨五入(round)
    hist = np.int32(np.around(hist_item))

    for x, y in enumerate(hist):
        cv2.line(h_result, (x, 0), (x, y), (255, 255, 255))
    # 上下方向地翻轉 array 的 item
    return np.flipud(h_result)

def backproject(feature_img, test_img):
    hsv_feature = cv2.cvtColor(feature_img, cv2.COLOR_BGR2HSV)
    hsv_target = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)

    # 計算兩張圖的 H-S histogram
    hist_feature = cv2.calcHist([hsv_feature], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 正規化 histogram 並且套用 backprojection
    cv2.normalize(hist_feature, hist_feature, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv_target], [0, 1], hist_feature, [0, 180, 0, 256], 1)

    # 用一個橢圓的 kernel 遮罩將 backprojection 圖補平平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # print(kernel)
    cv2.filter2D(dst, -1, kernel, dst)

    # 將結果做二元化
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    # 將二元化的結果 merge 成一張 3 channel 的圖
    thresh = cv2.merge((thresh, thresh, thresh))
    # 將 test image 與 遮罩 結果作交集
    res = cv2.bitwise_and(test_img, thresh)

    # 將三張圖的陣列資料做垂直堆疊
    res = np.vstack((test_img, thresh, res))

    w = int(res.shape[1] / 2.5)
    h = int(res.shape[0] / 2.5)
    # Resize 結果
    res_scaled = cv2.resize(res, (w, h))
    cv2.imshow('Backprojection', res_scaled)
    pass

def run(feature_path, test_path):
    feature_img = cv2.imread(feature_path)
    feature_gray = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
    test_img = cv2.imread(test_path)

    cv2.imshow('Feature Image', feature_img)
    cv2.imshow('Gray Image', feature_gray)
    while True:
        k = cv2.waitKey(0)
        if k == ord('1'):
            curve = hist_draw_curve(feature_img)
            cv2.imshow('RGB Histogram', curve)
        if k == ord('2'):
            curve = hist_draw_curve(feature_img, True)
            cv2.imshow('Normalized RGB Histogram', curve)
        elif k == ord('3'):
            lines = hist_draw_lines(feature_gray)
            cv2.imshow('Gray Histogram', lines)
            # 1. 用 matplotlib 計算 histogram 並畫出
            # plt.hist(feature_gray.ravel(), 256, [0, 256])
            # plt.show()
            # 2. 用 opencv 計算 histogram 再用 matplotlib 畫出
            # hist = cv2.calcHist([feature_gray], [0], None, [256], [0, 256])
            # plt.plot(hist, color = (0, 0, 0))
            # plt.xlim([0, 256])
            # plt.show()
        elif k == ord('4'):
            backproject(feature_img, test_img)
        elif k == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    pass