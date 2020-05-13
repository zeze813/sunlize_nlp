# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         手写数字判断
# Description:  
# Author:       SLZ
# Date:         2019/7/2
#-------------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.svm import SVC

'''
#加载并返回数字数据集（分类）;每个数据点是一个8x8的数字图像。
digits = datasets.load_digits()
print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[4]) #显示数字4
plt.show()
'''

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#加载图片数据
digits=datasets.load_digits()
# print(digits)

# 获取样本数量，并将图片数据格式化（要求所有图片的大小、像素点都是一致的 => 转换成为的向量大小是一致的）
n_samples=len(digits.images)
data=digits.images.reshape(n_samples,-1)
# print(data.shape)

#模型构建
model=SVC(gamma=0.001)

#使用二分之一的数据进行模型训练,取前一半数据训练，后一半数据测试
x_train=data[:int(n_samples / 2)]
x_test=data[int(n_samples / 2):]
y_train=digits.target[:int(n_samples / 2)]
y_test=digits.target[int(n_samples / 2):]
model.fit(x_train, y_train)

# 测试数据部分实际值和预测值获取
y_predicted=model.predict(x_test)

#metrics 模块包括评分函数，性能指标和成对指标以及距离计算。
print('模型效果为:{}'.format(metrics.classification_report(y_test,y_predicted)))
print('混淆矩阵为:\n{}'.format(metrics.confusion_matrix(y_test,y_predicted)))

# 进行图片展示
plt.figure(facecolor='gray', figsize=(12,5))
# 先画出5个预测失败的,把预测错的值的 x值 y值 和y的预测值取出
images_and_predictions = list(zip(digits.images[int(n_samples / 2):][y_test != y_predicted], y_test[y_test != y_predicted], y_predicted[y_test != y_predicted]))
##通过enumerate，分别拿出x值 y值 和y的预测值的前五个，并画图
for index,(image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')#把cmap中的灰度值与image矩阵对应，并填充
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))

## 再画出5个预测成功的
images_and_predictions = list(zip(digits.images[int(n_samples / 2):][y_test == y_predicted], y_test[y_test == y_predicted], y_predicted[y_test == y_predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
#     plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()



