import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

n_classes = 3
plot_colors = "bmy"
plot_step = 0.02

#加载数据
iris = load_iris()

for pairdix,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
	X = iris.data[:,pair]
	y = iris.target

	clf = DecisionTreeClassifier().fit(X,y)

	#绘制决策边界
	plt.subplot(2,3,pairdix+1)

	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	'''
	meshgrid 生成网格矩阵
	a = [1,2,3]
	b = [9,8,7]
	c, d = np.meshgrid(a,b)
	运行
	[[1 2 3]
	[1 2 3]
	[1 2 3]]
	----------
	[[9 9 9]
	[8 8 8]
	[7 7 7]]
	'''

	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
	#c_ 矩阵左右相加 r_上下相加
	Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)
	#绘制等高线
	cs = plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)

	plt.xlabel(iris.feature_names[pair[0]])
	plt.ylabel(iris.feature_names[pair[1]])
	plt.axis("tight")

	for i,color in zip(range(n_classes),plot_colors):
		idx = np.where( y == i)
		plt.scatter(X[idx,0],X[idx,1],c=color,label = iris.target_names[i],cmap = plt.cm.Paired)
	plt.axis('tight')

plt.suptitle('Decision surface of a decision tree using paired features')
plt.legend()
plt.show()