import os
import sys

if __name__ == '__main__':
    # 取出執行參數
    args = sys.argv[:]
    if len(args) == 2:
        # args[0] 為被 python 執行的檔案 (在此為 hello_world.py)
        path_to_file = args[1]

        import cv2
        if os.path.isfile(path_to_file):
            basename = os.path.basename(path_to_file)
            image = cv2.imread(path_to_file)
            # https://docs.opencv.org/3.3.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
            # 設定顯示視窗的標題, 與要顯示的陣列資料 (在此為影像資料)
            cv2.imshow(basename, image)
            # 0 : 等待直到按下鍵盤任一鍵, 若設定其他數字則單位為毫秒(ms)
            cv2.waitKey(0)
        else:
            print('Error ! 輸入路徑並非檔案 !')
    else:
        print('Error ! 請輸入正確參數格式. i.e. 執行 python hello_world.py PATH/TO/IMAGE')
