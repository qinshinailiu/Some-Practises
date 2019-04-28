import numpy as np
import operator

def createDataSet():
    #四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels
'''
基本原理
对于未知类别属性的数据中的每个点依次执行操作
1.计算已知类别数据集中的点，与当前点之间的距离
2.按照距离升序排序
3.选取与当前点距离最小的k个点
4.确定前k个点所在类别的出现频率
5.返回前k个点出现频率最高的类别作为当前点的预测分类

'''

'''
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点

'''
def classify0(inx,dataSet,labels,k):
	#获取数据集有多少行
	dataSetSize = dataSet.shape[0]
	#计算距离 利用tile 形成一个与训练集行数相同的矩阵
	diffMat = np.tile(inx,(dataSetSize,1)) - dataSet
	#print(diffMat)
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1)
	#以上操作就是求每个特征值差的平方，再相加
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	#选择距离最小的k个点
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

if __name__ == '__main__':
	group,labels = createDataSet()
	test = [100,15]
    #kNN分类
	test_class = classify0(test, group, labels, 3)
	print(test_class)