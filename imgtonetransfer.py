# import the necessary packages
import cv2
import numpy as np

def color_transfer(source, target):
	# 將影像從 BGR 轉換到 Lab, 並且利用浮點數來增加計算精細度.
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	def calculate_statistics(image):
		# 將 image 的三個 channel 分開並各自集合起來.
		l, a, b = cv2.split(image)
		# 計算這三個 channel 各自的平均數, 以及標準差.
		l_mean, a_mean, b_mean = l.mean(), a.mean(), b.mean()
		l_std, a_std, b_std = l.std(), a.std(), b.std()
		# 將各 channel 的統計資料傳出.
		return l_mean, l_std, a_mean, a_std, b_mean, b_std

	# 計算 source & target 的影像統計資料.
	l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src = calculate_statistics(source)
	l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar = calculate_statistics(target)
 
	# 將 target 的各 channel 分割, 並各自減去 target 的各 channel 平均數.
	l, a, b = cv2.split(target)
	l -= l_mean_tar
	a -= a_mean_tar
	b -= b_mean_tar
 
	# 依照標準差的比例(target/src)來調整 l,a,b
	l = (l_std_tar / l_std_src) * l
	a = (a_std_tar / a_std_src) * a
	b = (b_std_tar / b_std_src) * b
 
	# 將 Source 的各 channel 平均加回 target l,a,b
	l += l_mean_src
	a += a_mean_src
	b += b_mean_src
 
	# 邊界處理, 將值限制在0~255.
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
 
	# 將最終的 3 個 channel 組合回 per pixel, 並且將色彩空間轉回 BGR.
	result = cv2.merge([l, a, b])
	result = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	return result

def show_image(img, name='Unknow', downscale=True):
	# 降低顯示圖檔的解析度(如果超過1920x1080)
	cv2.namedWindow(name, 0)
	cv2.moveWindow(name, 0, 0)
	if (img.shape[0] > 1080 or img.shape[1] > 1920) and downscale:
		cv2.resizeWindow(name, int(img.shape[1]/4), int(img.shape[0]/4))
	else:
		cv2.resizeWindow(name, img.shape[1], img.shape[0])
	cv2.imshow(name, img)

def run(tar, src):
	source = cv2.imread(src)
	target = cv2.imread(tar)
	show_image(source, 'Source')
	show_image(target, 'Target')

	cv2.waitKey(0);
	result = color_transfer(source, target)
	show_image(result, 'Result')
	cv2.waitKey(0)