#coding=utf-8

import os
# 匯入 opencv 模組
import cv2

def window_management(win_name, w=0, h=0):
    # 0 : 不自動調整視窗大小, 1 : 自動調整視窗大小
    cv2.namedWindow(win_name, 0)

    # 第一個參數 : window name, 第二三參數 : 座標 x,y
    cv2.moveWindow(win_name, 0, 0)

    # 必須搭配 namedWindow 所指定的視窗, 且 AutoSize 不得為 1
    if w != 0 and h != 0:
        cv2.resizeWindow(win_name, int(w/2), int(h/2))

def run():
    path_to_file = 'HelloWorld.png'
    if os.path.isfile(path_to_file):
        basename = os.path.basename(path_to_file)

        image = cv2.imread(path_to_file)
        w, h = image.shape[1], image.shape[0]
        # https://docs.opencv.org/3.3.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
        # 設定顯示視窗的標題, 與要顯示的陣列資料 (在此為影像資料)

        # 控制顯示視窗大小位置
        # window_management(basename, w, h)

        cv2.imshow(basename, image)

        # 0 : 等待直到按下鍵盤任一鍵, 若設定其他數字則單位為毫秒(ms)
        cv2.waitKey(0)
        # 解構所有建立出來的 windows
        cv2.destroyAllWindows()
    else:
        print('Error ! 輸入路徑並非檔案 !')