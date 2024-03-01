# -*- coding: utf-8 -*-
# Time : 2024/2/23 11:53
# Author : chen
# Software: PyCharm
# File : 乳腺癌案例.py
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import StandardScaler#标准化特征缩放
from sklearn.preprocessing import MinMaxScaler#归一化特征缩放
from sklearn.model_selection import train_test_split#留出法
#导入乳腺癌本地数据集
data = pd.read_csv('breast-cancer-wisconsin.data',header=None)
#数据检视三板斧
#打印数据的基本信息
data.info()
#打印数据的前五行
print(data.head())
#打印数据的后五行
print(data.tail())
#打印数据的基本结构
print(data.describe())
'''数据分析'''
#分析的数据的异常值
#将数据当中的?改成1
#什么样的数据类型是适合机器学习的 ：整形  浮点型  object-->字符串类型适合机器学习吗
#将？改成1的思想：怎么解决   替换  replace()
data.replace('?',1,inplace=True)#将数据返回
data[6] = data[6].astype(int)
data.info()
#预测分类数据 二分类  多分类    恶性  良性   0   1
#将2改成0  4改成1
data[10] = data[10].map({2:0,4:1})
print(data.head(20))
#查看正样本和负样本是否均衡   过采样和欠采样
print(data[10].value_counts())
data_new = data[data[10] == 1].sample(458-241)
data = pd.concat([data_new,data],axis=0)
print(data[10].value_counts())
'''机器学习'''
#做机器学习需要拿出特征和标签
y = data.pop(10)
x = data
#对特征做标准化特征缩放处理
std = StandardScaler()
x = std.fit_transform(x)
#使用留出法划分数据 训练和测试占比7:3   随机洗牌
#为什么要对特征进行洗牌：
# np_perm = np.random.permutation(len(x))#打乱顺序
# x = x[np_perm]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,shuffle=True)
#引入逻辑回归  调Api写底层
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#创建逻辑模型
lo= LogisticRegression()
#惩罚因子/正则化系数 C
#网格搜索交叉验证
model = GridSearchCV(
    estimator=lo,
    param_grid={'C':[0.1,0.2,0.3,1.0,2.0,3.0,88,99,888]},
    cv=5
)
model.fit(train_x,train_y)
#打印最优参数
print(model.best_params_)
#打印最优得分
print(model.best_score_)
#重新创建逻辑模型
lo = LogisticRegression(C=model.best_params_['C'])
#训练/预测
lo.fit(train_x,train_y)
#预测测试数据
print(lo.predict(test_x))

#模型评测
#1.准确率  2.查全率  3.查准率  f1分数  混淆矩阵  分类报告  roc  auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(f'模型的准确率是{lo.score(test_x,test_y)}')
print(f'模型的混淆矩阵{confusion_matrix(test_y,lo.predict(test_x))}')
print(f'模型的分类报告{classification_report(test_y,lo.predict(test_x))}')
#打印预测的概率值
test_preict_proba = lo.predict_proba(test_x)
print(test_preict_proba)
#绘制roc曲线
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,th = roc_curve(test_y,test_preict_proba[:,1])
plt.plot(fpr,tpr,'r')
plt.show()
# 打印auc数值
print('auc数值',roc_auc_score(test_y,test_preict_proba[:,1]))


















































