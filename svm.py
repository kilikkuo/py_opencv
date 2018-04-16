import cv2 as cv
import numpy as np

CELL_SIZE=20
NUM_OF_BIN = 16

# 反曲斜
def deskew(img):
    # 關於矩, see 
    # [1] http://aishack.in/tutorials/image-moments/
    # [2] https://www.cnblogs.com/ronny/p/3985810.html
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()

    # 透過計算後的skew值, 設計仿射矩陣
    # 關於仿射矩陣的運算, see [1]
    # [1] https://blog.csdn.net/q123456789098/article/details/53330484
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * CELL_SIZE * skew],
                    [0, 1   , 0]])
    # 利用反向仿射矩陣矯正偏斜圖像
    img = cv.warpAffine(img, M, (CELL_SIZE, CELL_SIZE), flags=cv.WARP_INVERSE_MAP|cv.INTER_LINEAR)
    return img

#  Histogram of Oriented Gradients
def hog(img):
    # 利用 Soble 來計算 x 與 y 方向的一階梯度值
    # gx 主要凸顯垂直方向線條, gy 主要凸顯水平方向線條
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)

    # 計算 2D 陣列資料的幅值與徑度
    mag, ang = cv.cartToPolar(gx, gy)
    # 將徑度作 16 分的量化
    bins = np.int32(NUM_OF_BIN * ang / (2 * np.pi))
    
    # 將方向與幅度切割成四個 sub-square.
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

    # 以 mag_cells 值作為權重重新計算 bins 的出現次數, 即得到我們要的 hists,
    # 參考說明, https://blog.csdn.net/xlinsist/article/details/51346523
    hists = [np.bincount(b.ravel(), m.ravel(), NUM_OF_BIN) for b, m in zip(bin_cells, mag_cells)]
    # 共 16 x 4
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def run(train_ratio, test_ratio):
    # 共 50 行(0~9, 每一個數字各五行), 每一行有 100 個不同風格的字
    # 2000 x 1000
    img = cv.imread('digits.png')
    if img is None:
        raise Exception("we need the digits.png image from samples/data here !")

    # 將影像轉成灰階
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 將圖檔平均切割成 5000 個 cells, 每一個 cell 為 20 x 20.
    # 100 x 50 cells, 20 x 20 each.
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # 定義所有資料的前 X % 做為訓練用 Cells
    # 定義所有資料的後 Y % 做為測試用 Cells
    cell_cols = len(cells[0])
    len_of_train = int(cell_cols * train_ratio)
    len_of_test = cell_cols - int(cell_cols * test_ratio)
    # 印出變數的維度, 方便理解
    # print('len of train :{}'.format(len_of_train))
    # print('len of train :{}'.format(len_of_test))

    train_cells = [i[:len_of_train] for i in cells]
    test_cells = [i[len_of_test:] for i in cells]
    # 印出變數的維度, 方便理解
    # print('len of train_cells :{}'.format(len(train_cells)))
    # print('len oftest_cellstrain :{}'.format(len(test_cells)))
    # print('len of train_cells[0] :{}'.format(len(train_cells[0])))
    # print('len oftest_cellstrain[0] :{}'.format(len(test_cells[0])))

    # Training 
    deskewed = [map(deskew, row) for row in train_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]

    # 64 x 1 x len_of_train * cell_cols
    trainData = np.float32(hogdata).reshape(-1, 64)

    # 製作一個 response array, 數字由 0 ~ 9, 每個數字重複 len_of_train * 5 次
    # 並將其維度轉變成 1 x len_of_train * 5 * 10
    responses = np.repeat(np.arange(10), len_of_train * 5)[:, np.newaxis]
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setType(cv.ml.SVM_C_SVC)
    # Penalty multiplier, for more detail.
    # See https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine
    # svm.setC(2.67)
    # svm.setGamma(5.383)

    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')

    # 測試
    deskewed = [map(deskew,row) for row in test_cells]
    hogdata = [list(map(hog,row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, NUM_OF_BIN * 4)
    result = svm.predict(testData)[1]

    # 計算模型的正確率
    testResponses = np.repeat(np.arange(10), len(test_cells[0]) * 5)[:, np.newaxis]
    mask = result==testResponses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)