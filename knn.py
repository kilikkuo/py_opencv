import cv2
import numpy as np
import matplotlib.pyplot as plt

def random_points_knn():
    # 隨機產生 101 個二維訓練資料, 每一維度的值介於 0~100
    trainData = np.random.randint(0, 100,(101, 2)).astype(np.float32)
    # 隨機產生 101 個一維標記資料, 數值為 0 or 1
    data_labels = np.random.randint(0, 2,(101, 1)).astype(np.float32)

    # 將標記值為 0 所對應的訓練資料內設為紅色
    red = trainData[data_labels.ravel() == 0]
    plt.scatter(red[:,0], red[:,1], 80, 'r', 'o')
    # 將標記值為 1 所對應的訓練資料內設為藍色
    blue = trainData[data_labels.ravel() == 1]
    plt.scatter(blue[:,0], blue[:,1], 80, 'b', '^')

    # 隨機產生 3 個測試資料
    num_new = 3
    newcomer = np.random.randint(0, 100, (num_new, 2)).astype(np.float32)

    # 建立 knn 分類器
    knn = cv2.ml.KNearest_create()
    # 將訓練資料與標記丟入分類器訓練
    knn.train(trainData, cv2.ml.ROW_SAMPLE, data_labels)
    # 將測試資料丟入分類器, 得到分類結果
    ret, results, neighbours, dist = knn.findNearest(newcomer, 3)
    print("results: ", results,"\n")
    print("neighbours: ", neighbours,"\n")
    print("distances: ", dist)
    
    # 將三個測試資料的結果顯示出來, 並標記他們被分到哪一類(紅 or 藍)
    markers = ['*', 's', 'p']
    labels = ['red', 'blue']
    for i in range(num_new):
        res = results[i][0].astype(np.int32)
        color = labels[res]
        plt.scatter(newcomer[i][0], newcomer[i][1], c='g', marker=markers[i], label=color)
    plt.legend()
    plt.show()

def ocr_alphabet_knn():
    # Load the data, converters convert the letter to a number
    data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
                     converters= {0: lambda ch: ord(ch)-ord('A')})

    # 將資料切成兩部分(訓練 & 測試), 每一部分有 10000 筆資料.
    train, test = np.vsplit(data, 2)

    # 將訓練資料切割成兩部分, 第一部分是資料的標記, 第二部分為資料特徵
    responses, trainData = np.hsplit(train, [1])
    # 將測試資料切割成兩部分, 第一部分是正確標記, 第二部分為測試特徵
    labels, testData = np.hsplit(test,[1])

    # 建立 knn 分類器
    knn = cv2.ml.KNearest_create()
    # 將訓練資料的特徵與標記交與 knn 做訓練
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    # 將待測資料丟入 knn 計算每一個測試特徵屬於哪個標記.
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)

    # 將測試結果與測試資料的正確標記做精確度分析 
    correct = np.count_nonzero(result == labels)
    accuracy = correct / 10000.0 * 100
    print(' {} %'.format(accuracy))

def run(example):
    if example == 'points':
        random_points_knn()
        return
    elif example == 'alphabet':
        ocr_alphabet_knn()
        return
    assert False, '請指定參數 "points" or "alphabet"'