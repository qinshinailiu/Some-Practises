'''
项目案例1: 判定鱼类和非鱼类
项目概述

根据以下2个特征，将动物分成两类：鱼类和非鱼类。

特征: 1.不浮出水面是否可以生存 2.是否有脚蹼

开发流程

收集数据：可以使用任何方法
准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化
分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期
训练算法：构造树的数据结构
测试算法：使用决策树执行分类
使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义

'''
import operator
import matplotlib.pyplot as plt 
from math import log
from collections import Counter
import DecisionTreePlot as dtPlot

def createDataSet():
	dataSet = [[1, 1, 'yes'],
			[1, 1, 'yes'],
			[1, 0, 'no'],
			[0, 1, 'no'],
			[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def calcShannonEnt(dataSet):
    """
    Desc：
        calculate Shannon entropy -- 计算给定数据集的香农熵
    Args:
        dataSet -- 数据集
    Returns:
        shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    """
    #第一种实现方法
    #计算list长度，表示参与训练的数据量
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
    	currentLabel = featVec[-1]
    	if currentLabel not in labelCounts.keys():
    		labelCounts[currentLabel] = 0
    	labelCounts[currentLabel] +=1

    shannonEnt = 0.0
    for key in labelCounts:
    	#计算所有类别出现的频率
    	prob = float(labelCounts[key])/numEntries
    	shannonEnt -= prob *log(prob,2)

    '''
    第二种方法
    统计标签出现的次数
    label_count = Counter(data[-1] for data in dataset)
    计算概率
    probs = [p[1]/len(dataSet) for p in label_count.items()]
	计算香农熵
	shannonEnt = sum([-p * log(p,2) for p in probs])
    '''
    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet,index,value):
    """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    retDataSet = []
    for featVec in dataSet:
    	if featVec[index] == value:
    		reducedFeatVec = featVec[:index]
    		reducedFeatVec.extend(featVec[index+1:])
    		retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
	"""chooseBestFeatureToSplit(选择最好的特征)

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """
	numFeatures = len(dataSet[0])-1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain,bestFeature = 0.0,-1
	for i in range(numFeatures):
    	#获取对应特征下的所有数据
		featList = [example[i] for example in dataSet]
    	#去除重复值
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDateSet = splitDataSet(dataSet,i,value)
			prob = len(subDateSet)/float(len(dataSet))
			newEntropy = prob * calcShannonEnt(subDateSet)
    	#信息增益: 划分前后的信息变化
		infoGain = baseEntropy - newEntropy
		print('infoGain=',infoGain,'bestFeature=',i,baseEntropy,newEntropy)
		if( infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	"""majorityCnt(选择出现次数最多的一个结果)
	Args:
		classList label列的集合
	Returns:
		bestFeature 最优的特征列
	"""
	# -----------majorityCnt的第一种方式 start------------------------------------
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	# 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	# print 'sortedClassCount:', sortedClassCount
	return sortedClassCount[0][0]
	# -----------majorityCnt的第一种方式 end------------------------------------

	# # -----------majorityCnt的第二种方式 start------------------------------------
	# major_label = Counter(classList).most_common(1)[0]
	# return major_label
	# # -----------majorityCnt的第二种方式 end------------------------------------


def createTree(dataSet,labels):
	"""
	Desc:
		创建决策树
	Args:
		dataSet -- 要创建决策树的训练数据集
		labels -- 训练数据集中特征对应的含义的labels，不是目标变量
	Returns:
		myTree -- 创建完成的决策树
	"""
	classList = [example[-1] for example in dataSet]
	#第一个停止条件，所有类标签完全相同，直接返回该类标签
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	#第二个停止条件，用完了所有的特征，仍然不能将数据集划分成仅包含唯一类别的分组
	if len(dataSet) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	#初始化mytree
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		#求出剩余标签label
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)

	return myTree


def classify(inputTree,featLabels,testVec):
	"""
	Desc:
		对新数据进行分类
	Args:
		inputTree  -- 已经训练好的决策树模型
		featLabels -- Feature标签对应的名称，不是目标变量
		testVec    -- 测试输入的数据
	Returns:
		classLabel -- 分类的结果值，需要映射label才能知道名称
	"""
	#获取tree的根节点对于key的值
	firstStr = list(inputTree.keys())[0]
	#得到根节点对应的value
	secondDict = inputTree[firstStr]
	#判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么对树做分类
	featIndex = featLabels.index(firstStr)
	#测试数据,找到根结点对应的label位置
	key = testVec[featIndex]
	valueOfFeat = secondDict[key]
	print('+++',firstStr,'xxx',secondDict,'---',key,'>>>',valueOfFeat)
	#判断分支是否结束：判断valueOfFeat是否是dict类型
	if isinstance(valueOfFeat,dict):
		classLabel = classify(valueOfFeat,featLabels,testVec)
	else:
		classLabel = valueOfFeat
	return classLabel


def storeTree(inputTree, filename):
	"""
	Desc:
		将之前训练好的决策树模型存储起来，使用 pickle 模块
	Args:
		inputTree -- 以前训练好的决策树模型
		filename -- 要存储的名称
	Returns:
		None
	"""
	import pickle
	# -------------- 第一种方法 start --------------
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()
	# -------------- 第一种方法 end --------------

	# -------------- 第二种方法 start --------------
	with open(filename, 'wb') as fw:
		pickle.dump(inputTree, fw)
	# -------------- 第二种方法 start --------------


def grabTree(filename):
	"""
	Desc:
		将之前存储的决策树模型使用 pickle 模块 还原出来
	Args:
		filename -- 之前存储决策树模型的文件名
	Returns:
		pickle.load(fr) -- 将之前存储的决策树模型还原出来
	"""
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)

def fishTest():
	"""
	Desc:
		对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
	Args:
		None
	Returns:
		None
	"""
	# 1.创建数据和结果标签
	myDat, labels = createDataSet()
	# print(myDat, labels)

	# 计算label分类标签的香农熵
	# calcShannonEnt(myDat)

	# # 求第0列 为 1/0的列的数据集【排除第0列】
	# print('1---', splitDataSet(myDat, 0, 1))
	# print('0---', splitDataSet(myDat, 0, 0))

	# # 计算最好的信息增益的列
	# print(chooseBestFeatureToSplit(myDat))

	import copy
	myTree = createTree(myDat, copy.deepcopy(labels))
	print(myTree)
	# [1, 1]表示要取的分支上的节点位置，对应的结果值
	print(classify(myTree, labels, [1, 1]))

	# 画图可视化展现
	dtPlot.createPlot(myTree)
	#createPlot(myTree)

#预测隐形眼镜类型
def ContactLensesTest():
	fr = open('lenses.txt')
	#解析数据
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	#数据对应的labels
	lensesLabels = ['age','prescript','astigmatic','tearRate']
	lensesTree = createTree(lenses,lensesLabels)
	print(lensesTree)
	dtPlot.createPlot(lensesTree)


if __name__ == "__main__":
	#fishTest()
    ContactLensesTest()