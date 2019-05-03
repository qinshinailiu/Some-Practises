'''
项目案例1: 使用 Logistic 回归在简单数据集上的分类
项目概述

在一个简单的数据集上，采用梯度上升法找到 Logistic 回归分类器在此数据集上的最佳回归系数

开发流程

收集数据: 可以使用任何方法
准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳
分析数据: 画出决策边界
训练算法: 使用梯度上升找到最佳参数
测试算法: 使用 Logistic 回归进行分类
使用算法: 对简单数据集中数据进行分类
'''
import numpy as np  
import matplotlib.pyplot as plt 


def load_data_set():
	"""
	加载数据集
	:return:返回两个数组，普通数组 
		data_arr -- 原始数据的特征
		label_arr -- 原始数据的标签，也就是每条样本对应的类别
	"""
	data_arr = []
	label_arr = []
	f = open('TestSet.txt', 'r')
	for line in f.readlines():
		line_arr = line.strip().split()
		# 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
		data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
		label_arr.append(int(line_arr[2]))
	return data_arr, label_arr


def sigmoid(x):
	# 这里其实非常有必要解释一下，会出现的错误 RuntimeWarning: overflow encountered in exp
	# 这个错误在学习阶段虽然可以忽略，但是我们至少应该知道为什么
	# 这里是因为我们输入的有的 x 实在是太小了，比如 -6000之类的，那么计算一个数字 np.exp(6000)这个结果太大了，没法表示，所以就溢出了
	# 如果是计算 np.exp（-6000），这样虽然也会溢出，但是这是下溢，就是表示成零
	# 去网上搜了很多方法，比如 使用bigfloat这个库（我竟然没有安装成功，就不尝试了，反正应该是有用的
	return 1.0 / (1 + np.exp(-x))

def plotBestFit(dataArr,labelMat,weights):
	'''
	Desc:
		将我们得到的数据可视化展示出来
	Args:
		dataArr:样本数据的特征
		labelMat:样本数据的类别标签，即目标变量
		weights:回归系数
	Returns:
		None
	'''
	n = np.shape(dataArr)[0]
	xcord1 ,ycord1 = [],[]
	xcord2 ,ycord2 = [],[]
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = np.arange(-3.0,3.0,0.1)
	'''
 	首先理论上是这个样子的。
	dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
	w0*x0+w1*x1+w2*x2=f(x)
	x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
	所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
	'''
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()



# 正常的处理方案
# 两个参数：第一个参数==> dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
# 第二个参数==> classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat。
def gradAscent(dataMatIn,classLables):
	# 转化为矩阵[[1,1,2],[1,1,2]....]
	dataMatrix = np.mat(dataMatIn)
	# 转化为矩阵[[0,1,0,1,0,1.....]]，并转制[[0],[1],[0].....]
	# transpose() 行列转置函数
	# 将行向量转化为列向量   =>  矩阵的转置
	labelMat = np.mat(classLables).transpose() # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
	m,n = np.shape(dataMatrix)
	alpha = 0.001
	#迭代次数
	maxCycles = 500
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h)
		#print(error)
		weights = weights + alpha*dataMatrix.transpose()*error
	return np.array(weights)

# 随机梯度上升
# 梯度上升优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
# 随机梯度上升一次只用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
	m,n = np.shape(dataMatrix)
	alpha = 0.01
	weights = np.ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha*error*dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	m,n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.0001
			randIndex = int(np.random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex]-h
			weights = weights + alpha*error*dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

def testLR():
	#1 收集数据
	dataMat,labelMat = load_data_set()
	dataArr = np.array(dataMat)
	#weights = gradAscent(dataArr,labelMat)
	#weights = stocGradAscent0(dataArr,labelMat)
	weights = stocGradAscent1(dataArr,labelMat)
	plotBestFit(dataArr,labelMat,weights)



if __name__ == '__main__':
	testLR()