# coding=utf-8
import csv
import os
import pickle
import cPickle
from math import ceil
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import operator
import math

'''
函数说明：计算一个list变量的方差
传参：a(要计算的list变量)
返回值：一个float变量
'''
def var1(a):
    a_mean = np.mean(a)
    b = 0
    for i in a:
        b += (i - a_mean)*(i - a_mean)
    b = b / (len(a)-1)
    return b

# 加载数据集
test = pd.read_csv('test.csv')
test.index = test['id']
# 求出来先验概率
p_label_1 = len(test[test['label'] == 1]) * 1.0 / len(test)
p_label_0 = len(test[test['label'] == 0]) * 1.0 / len(test)
# 因为是二分类，接下来将test分成两部分，一部分是label为1，一部分是label为0
test_1 = test[test['label'] == 1]
test_0 = test[test['label'] == 0]

# 现在分析下其中一条样本是什么样本，假设我们用第一条样本来分析
# 定义两个变量，一个里面存放 这个样本是正样本的可能性，一个变量里面存放 这个样本是负样本的可能性
p_pos = (1.0*len(test_1)+1)/(len(test)+2)
p_neg = (1.0*len(test_0)+1)/(len(test)+2) # 之所以加1 2是因为为了“拉普拉斯修正”
for i in test.columns[1:-1]:
    # 先判断它是什么样的“属性”
    if (test[i].dtype == 'float64'):
        # 得到这个特征的属性值
        i_pr = test.ix[1,i]
        # 首先计算在test_1中的均值，方差
        mean_1 = np.mean(list(test_1[i]))
        var_1 = var1(list(test_1[i]))
        p_1 = 1.0 / (math.sqrt(2 * math.pi) * var_1) * math.exp(   -(i_pr-mean_1)*(i_pr-mean_1) / (2*var_1*var_1)      )
        p_pos *= p_1

        # 首先计算在test_0中的均值，方差
        mean_0 = np.mean(list(test_0[i]))
        var_0 = var1(list(test_0[i]))
        p_0 = 1.0 / (math.sqrt(2 * math.pi) * var_0) * math.exp(-(i_pr - mean_0) * (i_pr - mean_0) / (2 * var_0 * var_0))
        p_neg *= p_0
        # print(p_1)
        # print(p_0)

    else:
        # 得到这个特征的属性值
        i_pr = test.ix[1, i]
        # 先在 test_1 中，看看 i_pr 出现的频率
        p_1 = (len(test_1[test_1[i] == i_pr]) + 1) / (len(test_1) * 1.0 + len(test[i].unique())) #加了拉普拉斯修正
        # 再在 test_0 中，看看 i_pr 出现的频率
        p_0 = (len(test_0[test_0[i] == i_pr]) + 1) / (len(test_0) * 1.0 + len(test[i].unique()))
        p_pos *= p_1
        p_neg *= p_0

        # print(p_1)
        # print(p_0)

if(p_pos>p_neg):
    print("This is a postive sample")
elif(p_pos==p_neg):
    print("not sure")
else:
    print("This is a negative sample")















