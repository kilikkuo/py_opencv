#coding=utf-8

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from utils import show_image
import math

frame = None
track_window = (None, None, None, None)
roi_hist = None
timer = None
term_crit = None
dst = None
def run(path_to_video, method):
    global term_crit
    # 使用 opencv 的繪圖方式
    use_cv2draw = True

    fig, ax = plt.subplots()
    if not use_cv2draw:
        plt.subplots_adjust(bottom=0.20)

    # 新增重置按鈕的軸線圖
    axnext = plt.axes([0.2, 0.1, 0.18, 0.04])
    axradio = plt.axes([0.4, 0.04, 0.25, 0.12])
    # 連結按鈕軸線圖與按鈕
    btn_next = Button(axnext, "Next Frame", color="white", hovercolor='0.5')
    rbradio = RadioButtons(axradio, ('ROI Selected', 'Stop Tracking'), active=1)

    cap = cv2.VideoCapture(path_to_video)

    # 設定演算法中止條件, 最大迭代次數到 200 或是精確度收斂小於 0.05, 兩者達一即可.
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 0.05)

    if not use_cv2draw:
        # 建立一個 timer, 每 16 ms 呼叫一次
        timer = fig.canvas.new_timer(interval=16)

    def draw_trackwindow(window=None):
        # 劃出追蹤物件的視窗
        global frame
        global track_window
        global roi_hist
        global dst
        target_window = window if window else track_window

        # 如果 target_window 有 item 是 None, 或 沒有 frame 則 return.
        if not all(target_window) or frame is None:
            return

        target_frame = frame.copy()
        # 將目標追蹤方框畫在 frame 上.
        x, y, w, h = target_window
        img = cv2.rectangle(target_frame, (x, y), (x+w, y+h), 255, 3)
        if use_cv2draw:
            cv2.imshow('RealFrame', img)
            if dst is not None:
                cv2.imshow('BPFrame', dst)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title('Frame')
            fig.canvas.draw()

    def roi_selected(val):
        global frame
        global track_window
        global roi_hist
        if val == 'ROI Selected' and all(track_window):
            l, t, w, h = track_window
            # 將目標區域 ROI(Region Of Interest) 的圖像轉換色彩空間
            roi = frame[t:t+h, l:l+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 製作 H-S 的直方圖, 並將之正規化
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 252-35], [0, 180, 35, 252])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # 將目標追蹤方框畫在 frame 上
            draw_trackwindow()
            if not use_cv2draw:
                # 啟動 timer
                timer.start()
        else:
            if not use_cv2draw:
                print('Stopping timer ...')
                timer.stop()

    rbradio.on_clicked(roi_selected)

    def track_next_frame():
        global track_window
        global roi_hist
        global frame
        global term_crit
        global dst
        # 將目標 frame 轉換成 HSV, 並利用 roi_hist 來做出 backproject 圖表,
        # 最後一個參數 scale 設成 255 可以將圖當成 8bit 表示秀出
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 35, 252], 255)

        if method == 'mean':
            # 套用 meanShift 演算法得到新的 track window
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        else:
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # 將 CamShift 得到的
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # 將目標追蹤方框畫在 frame 上
        draw_trackwindow()

    def update_next_frame():
        global frame
        global track_window
        # 讀取下一張 frame
        ret, frame = cap.read()
        if ret == True:
            # 如果 track_window 的所有欄位皆為非 None 或非 0
            if all(track_window):
                track_next_frame()
            else:
                # track_window 尚未設定, 純粹顯示下張 frame
                if use_cv2draw:
                    cv2.imshow('Frame', frame)
                else:
                    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ax.set_title('Frame')

            if not use_cv2draw:
                fig.canvas.draw()
        return ret

    def on_next_frame(event):
        if not use_cv2draw:
            update_next_frame()
        else:
            ret = update_next_frame()
            while not all(track_window):
                print("After selecting a region, press any key to continue !!")
                key = cv2.waitKey() & 0xff
                if key == 27:
                    break
            while ret:
                # 等待每 16 ms
                key = cv2.waitKey(16) & 0xff
                if key == 27:
                    break
                ret = update_next_frame()

    if not use_cv2draw:
        # 設定 timer 啟動時的執行函數
        timer.add_callback(on_next_frame, None)
        # 當按鈕被按下之後, 擷取下一張 frame
        btn_next.on_clicked(on_next_frame)

    def onCVMouseOp(event, x, y, flags, parm):
        global track_window
        if all(track_window):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # 取得 mouse down 的座標
            track_window = x, y, None, None
        elif event == cv2.EVENT_LBUTTONUP:
            # 取得 mouse up 的座標, 計算出 track window 範圍, 並更新追蹤結果
            w = int(x - track_window[0])
            h = int(y - track_window[1])
            if w < 0:
                track_window = int(track_window[0] + w), int(track_window[1]), int(math.fabs(w)), h
            else:
                track_window = int(track_window[0]), int(track_window[1]), w, h
            if h < 0:
                track_window = int(track_window[0]), int(track_window[1] + h), track_window[2], int(math.fabs(h))
            else:
                track_window = int(track_window[0]), int(track_window[1]), track_window[2], h
            roi_selected("ROI Selected")
            print(track_window)
        elif event == cv2.EVENT_MOUSEMOVE:
            # 選取框框, 並把框框畫在圖上
            if None not in [track_window[0], track_window[1]]:
                temp_window = None
                w = int(x - track_window[0])
                h = int(y - track_window[1])
                if w < 0:
                    temp_window = int(x), int(track_window[1]), int(math.fabs(w)), h
                else:
                    temp_window = int(track_window[0]), int(track_window[1]), w, h
                if h < 0:
                    temp_window = temp_window[0], int(y), temp_window[2], int(math.fabs(h))
                else:
                    temp_window = temp_window[0], int(track_window[1]), temp_window[2], h
                draw_trackwindow(temp_window)
        pass

    if use_cv2draw:
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', onCVMouseOp)

    def onMouseOp(event):
        global track_window
        if all(track_window):
            return
        if event.name == "button_press_event":
            # 取得 mouse down 的座標
            track_window = event.xdata, event.ydata, None, None
        elif event.name == "button_release_event":
            # 取得 mouse up 的座標, 計算出 track window 範圍, 並更新追蹤結果
            w = int(event.xdata - track_window[0])
            h = int(event.ydata - track_window[1])
            if w < 0:
                track_window = int(track_window[0] + w), int(track_window[1]), int(math.fabs(w)), h
            else:
                track_window = int(track_window[0]), int(track_window[1]), w, h
            if h < 0:
                track_window = int(track_window[0]), int(track_window[1] + h), track_window[2], int(math.fabs(h))
            else:
                track_window = int(track_window[0]), int(track_window[1]), track_window[2], h
            draw_trackwindow()
        elif event.name == "motion_notify_event":
            # 選取框框, 並把框框畫在圖上
            if None not in [track_window[0], track_window[1]]:
                w = int(event.xdata - track_window[0])
                h = int(event.ydata - track_window[1])
                if w < 0:
                    temp_window = int(event.xdata), int(track_window[1]), int(math.fabs(w)), h
                else:
                    temp_window = int(track_window[0]), int(track_window[1]), w, h
                if h < 0:
                    temp_window = temp_window[0], int(event.ydata), temp_window[2], int(math.fabs(h))
                else:
                    temp_window = temp_window[0], int(track_window[1]), temp_window[2], h
                draw_trackwindow(temp_window)
        pass


    cid_down = None
    cid_up = None
    cid_motion = None

    if not use_cv2draw:
        # 在 figure 上綁訂 mouse 事件
        cid_down = fig.canvas.mpl_connect('button_press_event', onMouseOp)
        cid_up = fig.canvas.mpl_connect('button_release_event', onMouseOp)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', onMouseOp)
        fig.show()

    on_next_frame(None)

    if not use_cv2draw:
        plt.show()
        fig.canvas.mpl_disconnect(cid_down)
        fig.canvas.mpl_disconnect(cid_up)
        fig.canvas.mpl_disconnect(cid_motion)
    else:
        cv2.waitKey()

    cap.release()
