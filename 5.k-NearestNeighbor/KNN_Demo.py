import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets
# %matplotlib inline

n_neighbors = 3
iris = datasets.load_iris()
X = iris.data[:,:2] #绘图所以适用前俩
y = iris.target


h = .02 #网格中的步长
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

for weights in ['uniform','distance']:
	clf = neighbors.KNeighborsClassifier(n_neighbors,weights = weights)
	clf.fit(X,y)

	#绘制决策边界，为此，我们为每个分配一个颜色
	x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
	y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
	xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
						np.arange(y_min,y_max,h))
	Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])

	    # 将结果放入一个彩色图中
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练点
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title("3-Class classification (k = %i, weights = '%s')"
	% (n_neighbors, weights))

plt.show()