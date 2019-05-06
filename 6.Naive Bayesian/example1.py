'''
项目案例1: 屏蔽社区留言板的侮辱性言论
项目概述

构建一个快速过滤器来屏蔽在线社区留言板上的侮辱性言论。如果某条留言使用了负面或者侮辱性的语言，
那么就将该留言标识为内容不当。对此问题建立两个类别: 侮辱类和非侮辱类，使用 1 和 0 分别表示。
'''
import numpy as np

def load_data_set():
	posting_list = [
		['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
		['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'gar e'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
		['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
	return posting_list,class_vec

def create_vocab_list(data_set):
	"""
	获取所有单词的集合
	:param data_set: 数据集
	:return: 所有单词的集合(即不含重复元素的单词列表)
	"""
	vocab_set = set()  #新建一个无序的不重复元素序列
	for item in data_set:
		vocab_set = vocab_set | set(item)
	return list(vocab_set)

def set_of_words2vec(vocab_list,input_set):
	"""
	遍历查看该单词是否出现，出现该单词则将该单词置1
	:param vocab_list: 所有单词集合列表
	:param input_set: 输入数据集
	:return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
	"""
	result = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			result[vocab_list.index(word)] = 1
		else:
			pass
	return result

def bag_words2vec(vocab_list,input_set):
	result = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			result[vocab_list.index(word)] +=1
		else:
			print('the word:{} is not in my vocabulary'.format(word))
	return result

def _tran_naive_bayes(train_mat,train_category):
	"""
	朴素贝叶斯分类原版
	:param train_mat: 文件单词类型矩阵
		总的输入文本，大致是 [[0,1,0,1,1...], [], []]
	:param train_category: 文件对应的类别分类， [0, 1, 0],
		列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
	:return: 
	"""
	train_doc_num = len(train_mat)
	words_num = len(train_mat[0])
	# 因为侮辱性的被标记为了1， 所以只要把他们相加就可以得到侮辱性的有多少
	# 侮辱性文件的出现概率，即train_category中所有的1的个数，
	# 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
	pos_abusive = np.sum(train_category)/train_doc_num

	# 单词出现的次数
	p0num = np.zeros(words_num)
	p1num = np.zeros(words_num)

	#整个数据集单词出现的次数
	p0num_all = 0
	p1num_all = 0
	for i in range(train_doc_num):
		if train_category[i] == 1:
			p1num += train_mat[i]
			p1num_all += np.sum(train_mat[i])
		else:
			p0num += train_mat[i]
			p0num_all += np.sum(train_mat[i])
	# 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
	# 即 在1类别下，每个单词出现次数的占比
	p1vec = p1num / p1num_all
	# 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
	# 即 在0类别下，每个单词出现次数的占比
	p0vec = p0num / p0num_all
	return p0vec,p1vec,pos_abusive



def tran_naive_bayes(train_mat,train_category):
	"""
	朴素贝叶斯分类修正版
	1.避免其中一个概率为0
	2.避免下溢出
	:param train_mat: 文件单词类型矩阵
		总的输入文本，大致是 [[0,1,0,1,1...], [], []]
	:param train_category: 文件对应的类别分类， [0, 1, 0],
		列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
	:return: 
	"""
	train_doc_num = len(train_mat)
	words_num = len(train_mat[0])

	pos_abusive = np.sum(train_category)/train_doc_num

	p0num = np.ones(words_num)
	p1num = np.ones(words_num)

	p0num_all = 2.0
	p1num_all = 2.0

	for i in range(train_doc_num):
		#侮辱性
		if train_category[i] == 1:
			p1num += train_mat[i]
			p1num_all += np.sum(train_mat[i])
		else:
			p0num +=train_mat[i]
			p0num_all += np.sum(train_mat[i])

	#改成log函数
	p1vec = np.log(p1num/p1num_all)
	p0vec = np.log(p0num/p0num_all)

	return p0vec,p1vec,pos_abusive


def classify_naive_bayes(vec2classify,p0vec,p1vec,p_class1):
	"""
	使用算法：
		# 将乘法转换为加法
		乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
		加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
	:param vec2classify: 待测数据[0,1,1,1,1...]，即要分类的向量
	:param p0vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
	:param p1vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
	:param p_class1: 类别1，侮辱性文件的出现概率
	:return: 类别1 or 0
    """

	# 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
	# 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
	# 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
	# 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
	p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
	p0 = np.sum(vec2classify * p0vec) + np.log(1-p_class1)
	if p1>p0:
		return 1
	else:
		return 0 


def testing_naive_bayes():
	'''
	测试朴素贝叶斯算法
	'''
	#1 加载数据集
	list_post,list_classes = load_data_set()
	#2 创建单词集合
	vocab_list = create_vocab_list(list_post)

	#3.计算单词是否出现并创建数据矩阵
	train_mat = []
	for post_in in list_post:
		train_mat.append( set_of_words2vec(vocab_list,post_in) )

	#4.训练数据
	p0v,p1v,p_abusive = tran_naive_bayes(np.array(train_mat),np.array(list_classes))
	#5.测试数据
	test_one = ['love','my','dalmation']
	test_one_doc = np.array(set_of_words2vec(vocab_list,test_one))
	print('the result is: {}'.format(classify_naive_bayes(test_one_doc, p0v, p1v, p_abusive)))
	test_two = ['stupid', 'garbage']
	test_two_doc = np.array(set_of_words2vec(vocab_list, test_two))
	print('the result is: {}'.format(classify_naive_bayes(test_two_doc, p0v, p1v, p_abusive)))

if __name__ == '__main__':
	testing_naive_bayes()