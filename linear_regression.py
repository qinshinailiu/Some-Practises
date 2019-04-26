import numpy as np
import pandas as pd
#实现一元线性回归
#初始化数据
def init_data():
	#加载数据
	data = np.loadtxt('data.csv',delimiter = ',')
	return data
'''numpy的结构类似
	data = [ 
	  [1,2],
	  [4,1],
	  [10,12]
	]
'''
def linear_regression():
	#设置学习率
	learning_rate = 0.1
	#设置初始的权重w和截距离b
	init_w = 0
	init_b = 0
	#设置最大迭代次数
	init_num = 100
	data = init_data()
	[w,b] = optimzier(data,init_w,init_b,learning_rate,init_num)
	print(w,b)
	return w,b

def optimzier(data,init_w,init_b,learning_rate,init_num):
	w = init_w
	b = init_b
	for i in range(init_num):
		w,b = gradient_descent(w,b,data,learning_rate)
		if i%10 == 0:
			#compute_cost(w,b,data)
			print(i,compute_cost(w,b,data)) 
	return [w,b]

#定义损失函数
def compute_cost(w,b,data):
	total = 0
	x = data[:,0]
	y = data[:,1]
	total = (y-w*x-b)**2
	total = np.sum(total,axis=0)
	return total/len(data)

#定义梯度下降
def gradient_descent(w,b,data,learning_rate):
	w_gradient = 0
	b_gradient = 0
	N = float(len(data))

	for i in range(len(data)):
		x = data[i,0]
		y = data[i,1]

		w_gradient += (2/N)*x*(w*x+b-y)
		b_gradient += (2/N)*(w*x+b-y)

	new_w = w-learning_rate*w_gradient
	new_b = b-learning_rate*b_gradient
	return [new_w,new_b]

if __name__ == '__main__':
	linear_regression()



