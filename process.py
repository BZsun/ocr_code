import cv2
import numpy as np
from matplotlib import pyplot as plt

def transfer2binary(filename):
    '''
        获得二值化图像，且文本内容为黑底白字
    '''
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 127  
    max_value = 255
    _, binary_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)

    binary_image = cv2.bitwise_not(binary_image)

    # 显示二值化结果
    # cv2.imshow('Binary Image', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return binary_image

def split_tline_image(image, rect_box):
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    # 定位所有字符串轮廓的外接矩阵框
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    words = sorted(words,key=lambda s:s[0],reverse=False)
    for word in words:
        # 根据单个字符的外接矩形框筛选轮廓
        if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            splite_image = cv2.resize(splite_image,(32,32))
            word_images.append(splite_image)

    for i,j in enumerate(word_images):
        plt.subplot(1,4,i+1)
        plt.imshow(word_images[i],cmap='gray')
    plt.show()

    return words, word_images

if __name__ == '__main__':
    filename = './data/test/3299.jpg'
    image = transfer2binary(filename)
    word_bounds, word_images = split_tline_image(image, rect_box=(2,2))
