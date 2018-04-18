import cv2 as cv
import time
import numpy as np

def run(image_path):
	# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
	# 參考並修改自 https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet/deploy.prototxt
	prototxt = './dnn/bvlc_googlenet.prototxt'
	# https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
	caffemodel = './dnn/bvlc_googlenet.caffemodel'
	# https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh
	# 透過下載得到的 tar.gz, 解壓縮後可得
	synsetwords = './dnn/synset_words.txt'

	image = cv.imread(image_path)

	# 載入分類標籤
	rows = open(synsetwords).read().strip().split("\n")
	# 找出第一個空格之後並出現在第一個逗號之前的字串
	classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

	# 根據 Deploy 的 Prototxt, 得知訓練後的 CNN model 需要輸入資料的維度為
	# (1, 3, 224, 224)
	# 透過訓練時的 Prototxt, 可以得知訓練圖集的 Pixel RGB 平均值為 (104, 117, 123)
	# https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt
	# 做 Mean subtraction 的目的 : 改善對抗光線改變造成的影響
	blob = cv.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
	
	# 載入網路
	net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)

	# 將 blob 輸入至訓練網路, 進行 forward-pass, 作分類
	net.setInput(blob)
	start = time.time()
	predicteds = net.forward()
	print("Cassification took {:.4} seconds".format(time.time() - start))

	# 找出預測結果中機率最大的前 3 名的索引
	# np.argsort 結果為升冪(ascending)排列
	indices = np.argsort(predicteds[0])[::-1][:3]

	# 顯示前三名
	for (i, idx) in enumerate(indices):
		cls_name = classes[idx]
		prob = predicteds[0][idx]
		if i == 0:
			text = "Label: {}, Probability: {:.4f}%".format(cls_name, prob * 100)
			cv.putText(image, text, (5, 15),  cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
		print("[{}] Label: {:10}, Probability: {:.4f}".format(i + 1, cls_name, prob * 100))

	cv.imshow("Image", image)
	cv.waitKey(0)