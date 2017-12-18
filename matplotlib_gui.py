#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def draw_figures():
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
    # https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
    fig, ax = plt.subplots()
    ax.set_facecolor('green')
    ax.set_title('Wave')

    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html#matplotlib.pyplot.subplots_adjust
    # 設定 subplot 的 layout
    plt.subplots_adjust(left=0.25, bottom=0.25)

    Xs = np.arange(0.0, 1.0, 0.001)
    Ys = np.sin(Xs * 5 * 2 * np.pi)
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
    l, = ax.plot(Xs, Ys, lw=2, color='red')
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axis.html#matplotlib.axes.Axes.axis
    # 設定初始座標資訊 x 座標起點,終點, y 座標起點,終點
    ax.axis([0, 1, -1, 1])

    # 設定軸色
    axcolor = 'lightgoldenrodyellow'
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes
    # 新增拖曳槓的軸線圖, left/bottom/width/height
    axmin = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    # https://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Slider
    # 連結對應的軸線, 設定最小值 1, 最大值 10, 初始值 5
    smin = Slider(axmin, 'Min', 1, 10, valinit=5)

    def update(val):
        new_ydata = np.sin(Xs * val * 2 * np.pi)
        l.set_ydata(new_ydata)
        # https://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle
        fig.canvas.draw_idle()
    # 設定 Slide 值變動時的 callback 函數
    smin.on_changed(update)

    # 新增重置按鈕的軸線圖
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # 連結按鈕軸線圖與按鈕
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.5')
    def reset_slider(event):
        smin.reset()
    # 當按鈕被按下之後, 重置 Slider 數值
    button.on_clicked(reset_slider)

    # 新增換色的軸線圖
    radioax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    # 設計一個 RadioButton 來連結換色圖
    # https://matplotlib.org/api/widgets_api.html#matplotlib.widgets.RadioButtons
    # active : 起始顏色的 index
    radio = RadioButtons(radioax, ('red', 'blue', 'black'), active=0)
    def change_color(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(change_color)

    plt.show()

def bgr_to_rgb(input_img):
    # OpenCV 的預設影像色彩順序為 BGR, 而 Matplotlib 是 RGB.
    import cv2
    return cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

def draw_images_NG():
    import cv2
    img = '101.JPG'
    img_data = cv2.imread(img)
    plt.title('Show cv2 data(BGR) by matplotlib API')
    plt.imshow(img_data)
    # 將 x 軸座標細節清除
    plt.xticks([])
    # 將 y 軸座標細節清除
    plt.yticks([])
    plt.show()

def draw_images_OK():
    import cv2
    img = '101.JPG'
    img_data = cv2.imread(img)
    plt.title('Convert cv2 data(BGR) to (RGB)')
    plt.imshow(bgr_to_rgb(img_data))
    # 標示 X 軸在[100, 200, 500, 1000] 的位置
    plt.xticks([100, 200, 500, 1000])
    plt.show()

def run():
    draw_figures()
    draw_images_NG()
    draw_images_OK()