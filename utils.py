#coding=utf-8

import cv2
import matplotlib.pyplot as plt

def show_image(image, title='image', wait=True, millisec=0):
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(millisec)
        cv2.destroyAllWindows()

# https://matplotlib.org/users/customizing.html
def set_plt_autolayout(auto):
    plt.rcParams["figure.autolayout"] = auto

# https://matplotlib.org/users/customizing.html
def update_plt_window_size(w, h):
    # w, h in inches
    # Get current size
    fig_size = plt.rcParams["figure.figsize"]
    print("Current size: {}".format(fig_size))
    fig_size[0] = w
    fig_size[1] = h
    plt.rcParams["figure.figsize"] = fig_size