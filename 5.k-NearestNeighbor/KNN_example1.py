'''
项目案例1: 优化约会网站的配对效果
项目概述

海伦使用约会网站寻找约会对象。经过一段时间之后，她发现曾交往过三种类型的人:

不喜欢的人
魅力一般的人
极具魅力的人
她希望：

工作日与魅力一般的人约会
周末与极具魅力的人约会
不喜欢的人则直接排除掉
现在她收集到了一些约会网站未曾记录的数据信息，这更有助于匹配对象的归类。

开发流程

收集数据：提供文本文件
准备数据：使用 Python 解析文本文件
分析数据：使用 Matplotlib 画二维散点图
训练算法：此步骤不适用于 k-近邻算法
测试算法：使用海伦提供的部分数据作为测试样本。
测试样本和非测试样本的区别在于：
测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

---------------------------------
收集数据：提供文本文件
海伦把这些约会对象的数据存放在文本文件 datingTestSet2.txt 中，总共有 1000 行。海伦约会的对象主要包含以下 3 种特征：

每年获得的飞行常客里程数
玩视频游戏所耗时间百分比
每周消费的冰淇淋公升数
文本文件数据格式如下：

40920   8.326976    0.953952    3  
14488   7.153469    1.673904    2  
26052   1.441871    0.805124    1  
75136   13.147394   0.428964    1  
38344   1.669788    0.134296    1
'''
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt 
import numpy as np 
#%matplotlib inline


def file2matrix(filename):
	"""
	Desc:
		导入训练数据
	parameters:
		filename:数据文件路径
	return:
		数据矩阵 returnMat 和对应类别classLabelVector
	"""
	#可以用Pd的方法
	fr = open(filename)
	#获得文件的数据行行数
	numberOfLines = len(fr.readlines())
	#生成对应的空矩阵
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		#str.strip([chars]) 返回移除字符串头尾指定的字符生成的新字符串
		line = line.strip()
		#以 '\t'切割字符串
		listFromLine = line.split('\t')
		#每列的属性数据
		returnMat[index,:] = listFromLine[0:3]
		#每列的类别数据，也就是label
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

"""
函数说明：可视化数据

Parameters:
	datingDateMat - 特征矩阵
	datingLabels  - 分类label
Returns:
	无
"""

def showdatas(datingDataMat,datingLabels):
	fig,axs = plt.subplots(nrows = 2,ncols = 2,sharex = False,sharey = False,figsize = (13,8))
	numberOfLables = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')
	#画散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha = 0.5)
    #设置标题,x轴label,y轴label
	axs0_title_text = axs[0][0].set_title('Air Mileage and Video Game')
	axs0_xlabel_text = axs[0][0].set_xlabel('Air Mileage')
	axs0_ylabel_text = axs[0][0].set_ylabel('Video Game')
	plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title('Air Mileage and Icecream Consumption')
	axs1_xlabel_text = axs[0][1].set_xlabel('Air Mileage')
	axs1_ylabel_text = axs[0][1].set_ylabel('Icecream Consumption')
	plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title('Video Game and Icecream Consumption')
	axs2_xlabel_text = axs[1][0].set_xlabel('Video Game')
	axs2_ylabel_text = axs[1][0].set_ylabel('Icecream Consumption')
	plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

	#设置图例
	didntLike = mlines.Line2D([],[],color = 'black',marker='.',markersize=6,label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')

	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])

	plt.show()

def autoNorm(dataSet):
	"""
	Desc:
		归一化特征值，消除特征之间量级不同导致的影响
	parameter:
		dataSet:数据集
	return:
		归一化后的数据集

	归一化公式:
		Y = (X - Xmin)/(Xmax - Xmin)
		其中的min和max分别是数据集中的最小特征值和最大特征值
		该函数可以将数字特征转化为0到1的区间
	"""
	#计算每种属性的最大值，最小值，范围
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#极差
	ranges = maxVals - minVals
	normDataSet = np.zeros(shape(dataSet))
	m = dataSet.shape[0]
	#tile  m行1次
	normDataSet = dataSet - np.tile(minVals,(m,1))
	# 将最小值之差除以范围组成的举证
	normDataSet = normDataSet/np.tile(ranges,(m,1))
	return normDataSet,ranges,minVals

if __name__ == '__main__':
	filename = 'datingTestSet2.txt'

	datingDataMat,datingLabels = file2matrix(filename)
	showdatas(datingDataMat,datingLabels)