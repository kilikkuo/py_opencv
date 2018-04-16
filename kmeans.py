import cv2 as cv
import numpy as np

def run(img_path, k):
    img = cv.imread(img_path)
    # 將影像維度從 W, H, 3, 轉成 WxH, 3
    pixels = np.float32(img.reshape((-1,3)))

    # TERM_CRITERIA_MAX_ITER : 最多幾次疊代後即停止
    # TERM_CRITERIA_EPS : 抵達特定精確度後即停止疊代
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # ret : 每個 pixel 與所屬群的中心點之距離平方之和
    # lable : 每個 pixel 被分與的群編號
    # center : 各群中心點
    # attempts : 測試 X 次不同的初始 label, 選取結果最好的一次當最後結果.
    ret, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    # 將 labels 結果轉換成一維陣列
    flatterned = labels.flatten()
    # 將 label 後的結果轉換成對應的 center (顏色)
    quantized = centers[flatterned]
    # 轉換成影像原始維度
    result = quantized.reshape((img.shape))
    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()