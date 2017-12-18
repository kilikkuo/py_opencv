#coding=utf-8

import cv2
import numpy
from utils import show_image

# OpenCV 支援的色彩空間 enumeration
# https://docs.opencv.org/3.3.0/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0

def to_gray(image):
    # Convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def to_hsv(image):
    # Convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def bgr_to_xyz_rec709_d65(image):
    # Convert to xyz
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    return xyz_image

def rec709_d65_xyz_to_bgr(image):
    # Convert to bgr
    bgr_image = cv2.cvtColor(image, cv2.COLOR_XYZ2BGR)
    return bgr_image

def extract_green_in_hsv(image):
    # 在 BGR 色彩空間中定義綠色
    bgr_green = numpy.uint8([[[0, 255, 0]]])
    # 將 BGR 色彩空間中的綠色轉換至 HSV 色彩空間
    hsv_green = cv2.cvtColor(bgr_green, cv2.COLOR_BGR2HSV)
    hue_green = hsv_green[0][0][0]

    # 建立兩個不同的綠色(HSV)
    lower_green = numpy.uint8([hue_green - 10, 100, 100])
    upper_green = numpy.uint8([hue_green + 10, 255, 255])

    # 製造一個遮罩, 過濾的值介於上述兩個綠色之間
    mask = cv2.inRange(image, lower_green, upper_green)
    show_image(mask, 'Mask', False)

    # 執行 bitwise operation, 將遮罩與圖交集.
    result = cv2.bitwise_and(image, image, mask= mask)
    rgb_result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    show_image(rgb_result, title='Result')
    pass

def run(sample_img):
    img_data = cv2.imread(sample_img)
    show_image(img_data, 'Original', False)

    gray_img = to_gray(img_data)
    show_image(gray_img, 'Gray')

    hsv_img = to_hsv(img_data)
    show_image(hsv_img, 'HSV')

    xyz_img = bgr_to_xyz_rec709_d65(img_data)
    show_image(xyz_img, 'BGR to XYZ')
    bgr_img = rec709_d65_xyz_to_bgr(xyz_img)
    show_image(bgr_img, 'XYZ to BGR')

    extract_green_in_hsv(hsv_img)


# def draw_figures(original, blurred):
#     # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
#     # https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
#     fig, axs = plt.subplots(2)

#     ax[0].set_title('Original')
#     ax[1].set_title('Averaged')

#     # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html#matplotlib.pyplot.subplots_adjust
#     # 設定 subplot 的 layout
#     plt.subplots_adjust(bottom=0.25)

#     # 設定軸色
#     axcolor = 'lightgoldenrodyellow'
#     # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes
#     # 新增拖曳槓的軸線圖, left/bottom/width/height
#     axhue = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
#     axsat = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
#     axval = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

#     # https://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Slider
#     # 連結對應的軸線, 設定最小值 1, 最大值 10, 初始值 5
#     shue = Slider(axhue, 'Hue', 0, 360, valinit=180)
#     ssat = Slider(axsat, 'Sat', 0, 255, valinit=255)
#     sval = Slider(axval, 'Val', 0, 255, valinit=255)

#     def update_hsv(val):
#         new_ydata = np.sin(Xs * val * 2 * np.pi)
#         l.set_ydata(new_ydata)
#         # https://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle
#         fig.canvas.draw_idle()
#     # 設定 Slide 值變動時的 callback 函數
#     shue.on_changed(update_hsv)
#     ssat.on_changed(update_hsv)
#     sval.on_changed(update_hsv)