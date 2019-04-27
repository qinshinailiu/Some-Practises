import numpy as np 
import pandas as pd 
import time
#引入datasets数据集
from sklearn import datasets
#引入线性模型
from sklearn.linear_model import LinearRegression
#数据集划分，分成训练集和测试集
from sklearn.model_selection import train_test_split

boston = datasets.load_boston() #载入波士顿房价
X = boston.data #特征值
y = boston.target	  #样本label

print("X.shape:{},y.shape:{}".format(X.shape,y.shape))
print('boston.feature_name:{}'.format(boston.feature_names))
'''
test_size：
　　float-获得多大比重的测试样本 （默认：0.25）
　　int - 获得多少个测试样本

train_size: 同test_size

random_state:
　　int - 随机种子（种子固定，实验可复现）
　　
shuffle - 是否在分割之前对数据进行洗牌（默认True）

返回
---
分割后的列表，长度=2*len(arrays), 
　　(train-test split)
'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

model = LinearRegression()
start = time.process_time()
model.fit(X_train,y_train)
# 为模型进行打分
train_score = model.score(X_train, y_train) # 线性回归：R square； 分类问题： acc
sv_score = model.score(X_test,y_test)


print('time used:{0:6f};train_score:{1:6f},sv_score:{2:6f}'.format((time.process_time()-start),train_score,sv_score))
# 模型预测
#model.predict(X_test)

# 获得这个模型的参数
#model.get_params()