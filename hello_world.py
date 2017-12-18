#coding=utf-8

import os

def run():
    path_to_file = 'HelloWorld.png'
    if os.path.isfile(path_to_file):
        basename = os.path.basename(path_to_file)
        # 匯入 opencv 模組
        import cv2
        image = cv2.imread(path_to_file)
        # https://docs.opencv.org/3.3.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
        # 設定顯示視窗的標題, 與要顯示的陣列資料 (在此為影像資料)
        cv2.imshow(basename, image)
        # 0 : 等待直到按下鍵盤任一鍵, 若設定其他數字則單位為毫秒(ms)
        cv2.waitKey(0)
        # 解構所有建立出來的 windows
        cv2.destroyAllWindows()
    else:
        print('Error ! 輸入路徑並非檔案 !')