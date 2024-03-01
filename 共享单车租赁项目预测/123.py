# -*- coding: utf-8 -*-
# Time : 2023/12/13 20:57
# Author : chen
# Software: PyCharm
# File : 共享单车代码实现.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime,time
import calendar#年历
import warnings
warnings.filterwarnings('ignore')
#读取共享单车数据集
datas = pd.read_csv('train.csv')
# 查看样本的所有数据，能够查看数据字段进行分析
print(datas.head())#默认查看数据的前五行
datas.info()#查看数据的基本信息
print(datas.describe())#查看数据的基本分布情况
print('==========时间处理===========')
#数据字段类型转换处理   datetime  ' 2011-01-01 00:00:00'
#转换时间组==>为了方便提取数据 组成一个新的数据表
# strftime() ==>时间组转换成时间串
# t = datetime.datetime.strftime()
# t = datetime.datetime.strptime('2011-01-01 00:00:00','%Y-%m-%d %H:%M:%S')
# print(t)
# t = datetime.datetime.fromisoformat('2011-01-01 00:00:00')
# print(t)
# print(t.year)#提取年
# print(t.month)#提取月份
# print(calendar.month_name[t.month])#提取一月份对应的英文单词/月份明细
# print(t.weekday())#想打印当天是周几对应的英文单词  5==>周六
#获取月份信息
def get_month(mt):
    #返回时间的datetime  时间组形式
    month_obj = datetime.datetime.fromisoformat(mt)
    #返回datetime中读取的月份信息
    return month_obj
#获取天的信息
def get_day(da):
    day_obj = datetime.datetime.fromisoformat(da).day
    return day_obj
#获取小时信息
def get_hour(ho):
    hour_obj = datetime.datetime.fromisoformat(ho).hour
    return hour_obj
#获取对应的月份名字
def get_month_name(ma):
    # 返回时间的datetime
    month_obj = datetime.datetime.fromisoformat(ma)
    month_value = month_obj.month
    return calendar.month_name[month_value]
#获取对应的星期名字
def get_weekday_name(dn):
    # 返回时间的datetime
    week_day_obj = datetime.datetime.fromisoformat(dn)
    weekday_value = week_day_obj.weekday()
    return calendar.day_name[weekday_value]
# 目的是将月份  天  小时添加到我们的datas里面中
datas['month'] = datas['datetime'].map(get_month)
datas['day'] = datas['datetime'].map(get_day)
datas['hour'] = datas['datetime'].map(get_hour)
#打印数据检查数据是否填入成功
datas.info()
#对应添加月份名字和星期名字的字段
datas['month_name'] = datas['datetime'].map(get_month_name)
datas['day_name'] = datas['datetime'].map(get_weekday_name)
#再次查看数据
datas.info()

#接下来可以出图/绘制图像  点线图
sns.pointplot(
    data=datas,
    x='hour',
    y='count'#骑行量

)
plt.show()
# sns.pointplot(
#     data=datas,
#     x='month',
#     y='count'
# )
# plt.show()
#图像观察
# 根据小时用车辆对数据进行分析
# 通过图像可以看出，
# 0-6点用车人数少，7-10点用车人数多，11-15为低谷，
# 16-20点为高峰，21-24用车人数少
# 数据分层 1.可以简化模型运算  2.更好的对数据分布情况进行分析计算
def hour_type(hour):
    if hour>=0 and hour<=6:
        return 1
    elif hour>=7 and hour<=10:
        return 2
    elif hour>=11 and hour<=15:
        return 3
    elif hour>=16 and hour<=20:
        return 4
    else :return 5

# 创建一个新的字段，用于表示用车时间段
datas["hour_type"]=datas.hour.map(hour_type)
#再次查看数据
datas.info()
#绘制图像
sns.pointplot(data=datas,x='hour',y='count')#y='count'#骑行量
plt.show()
sns.pointplot(data=datas,x='hour_type',y='count')#y='count'#骑行量
plt.show()
#大家现在观察一下 一次只能表示一个事件  要表达多个事件怎么办？？
#多花板绘制  通过季节来绘制箱型图==》能用来做对数据是否有噪音进行查看
# 三、数据降噪
# 对数据是否有噪音进行查看
# 只针对训练集进行降噪，为了更好拟合数据，让模型进度更优
# 因为测试集就代表未来的数据信息，而未来的数据信息中必定包含噪音，所以不对测试集进行降噪处理
#fig==>画板
#axes==>画板坐标
#多画板绘制 相当于是绘制子图
# nrows==>绘制多少行子图
# ncols==>绘制多少列子图
# fig ==》对应的是画板
# axes ==》对应的是坐标
fig,axes = plt.subplots(nrows=1,ncols=2)
sns.boxplot(
    data=datas,
    y='count',
    ax=axes[0]
)
sns.boxplot(
    data=datas,
    x='season',#季节
    y ='count',
    ax=axes[1]
)
plt.show()
#通过箱型图我们观察到有很多异常值 所以我们要把它去掉
#获取骑行量的数据  去除骑行量异常值
'''
iloc==>需要使用的是字段的索引或行索引对数据进行切分取值iloc[3,11]3行11列
loc==>需要使用的是字段名或行名对数据进行切分取值loc[:,['count']]
'''
get_counts = datas.loc[:,['count']]
print(get_counts)
# 查找噪音点（数据点-均值超过该数据标准差的三倍视为噪音），
# 并计算数量
#骑行量的均值
get_count_mean =  np.mean(get_counts)
#骑行量的标准差
get_count_std = np.std(get_counts)
#判断噪声的条件   数据点  -  均值 > 标准差的3倍
#计算每个骑行量与平均值之间的距离值
dist_count = get_counts - get_count_mean
#将距离值和3倍标准差进行比较，大于的就是噪声  否则就是正常值
noise_opt = dist_count > 3 * get_count_std
print(noise_opt)
#拿出异常值的数据  也就是判断异常值的数据  异常数据为true
noise_opt_val = noise_opt.values.flatten()  #也就是过滤
print(noise_opt_val)#出来的单独是一个列表
#接下来也就是把它作为一个筛选项  作为独一的把符合我们要求的
# 非异常数据保存  异常数据给她干掉
#也就是我们要提取非异常值 异常值我们不会要
print(~noise_opt_val)
'''
从数据当中获取非异常数据
~==>对数据进行取反
'''
#真实删除噪声  loc这个方法就是去true  所以要取反
get_datas_good = datas.loc[~noise_opt_val,:]
get_datas_good.info()

#项目进行到一半了 哈哈哈

#判断数据的相关性 热力图==>计算皮尔逊系数
# 需要把所有数据都进行计算吗  不需要  只需要有用的
#绘制热力图
heat_map_list = ['count', 'weather', 'temp',
                 'atemp', 'windspeed',  'casual', 'registered']

#分析天气跟骑行量相关的
#将符合热力图绘制的字段信息取出
dd_good_keep = get_datas_good.loc[:,heat_map_list]
#计算皮尔逊系数
dd_good_keep_corr = dd_good_keep.corr()
sns.heatmap(
    data=dd_good_keep_corr,
    annot=True
)
plt.show()
#观察字段跟骑行量相关性很强
#整个天气对骑行量影响很大  也就是逆向的
#影响最小的就是风速


#分析不同季节对骑行量的影响/分析季节对用车辆的影响
#按照‘season’字段进行分组,并得到分组后字段的平均值每个季节的信息
season_gb = get_datas_good.groupby(['season'])['count'].mean()
print('123',season_gb)
#重置索引
season_gb_good = season_gb.reset_index()
print(season_gb_good)
#绘制柱状图
sns.barplot(
    data=season_gb_good,
    x='season',
    y='count',
    hue='season'
)
plt.show()
#按照小时和季节hour,season字段进行分组  不同影响
hour_season = get_datas_good.groupby(['hour','season'])['count'].mean().reset_index()
print(hour_season)
sns.barplot(
    data=hour_season,
    x='hour',
    y='count',
    hue='season'
)
plt.show()
'''
数据分析操作
'''
datas.info()
#特征工程  对应处理的是x数据
#删除冗余字段 datetime  hour  month
#重复表达一个功能
get_datas_good.drop('datetime',inplace=True,axis=1)#axes=1按照 按照列删 0相反
get_datas_good.drop('hour',inplace=True,axis=1)#axes=1按照 按照列删 0相反
get_datas_good.drop('month',inplace=True,axis=1)#axes=1按照 按照列删 0相反

#展示数据
get_datas_good.info()
# 做完这些分析，准备做特征工程，比如归一化，缺失值填充，构造新的特征等。
#数据独热处理  ==》升纬   可能会过拟合
'''
#离散型：人为定义的数据   都为自然数  春夏秋冬 男女其他  一般会进行独热
#连续型：在某一个区间内  任意值[35.5-37.2]           一般会标准化
'''
#对季节进行独热  整数分为正整数和负整数
#只对关键性数据进行处理  防止过拟合
get_datas_good = pd.get_dummies(data=get_datas_good,columns=['season','weather','hour_type','month_name','day_name'])
get_datas_good.info()
#特征缩放  标准化
from sklearn.preprocessing import StandardScaler

std_list = ['temp', 'atemp', 'windspeed', 'humidity', 'casual', 'registered']
for i in std_list:#获取字段名
    #取一个字段的全部数据
    # get_datas_good[i]==>取出来的东西是一维的数据【1,2,3,4】
    #标准化需要的是二维的数据例如：get_datas_good[[i]]
    std = StandardScaler()
    #一定要返回原来的表
    get_datas_good[i] = std.fit_transform(get_datas_good[[i]])
#展示
get_datas_good.info()
'''========机器学习========='''
#准备特征矩阵和标签矩阵
y = get_datas_good.pop('count')#使用pop方法将指定的数据取出  原始数据集当中就没有了  作为返回值返回给变量
x = get_datas_good
print('特征',x)
#观察数据看能不能直接做机器学习  是可以的
#类型是dateframe 效率不高   数据类型转换为矩阵类型的数据  矩阵类型的数据更适合机器学习  减少负担
X_arr = np.array(x)
y_arr = np.array(y)
# 导包
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split,GridSearchCV
#切分数据集
train_x,test_x,train_y,test_y = train_test_split(X_arr,y_arr,test_size=0.2)
#调用岭回归  需要确定正则化系数
rd = Ridge()
#网格搜索交叉验证  确定系数
param_grid = {'alpha':[0.1,0.2,0.3,1,2,3]}
model = GridSearchCV(rd,param_grid=param_grid,cv=5)
#训练模型-==》找到最优的参数
model.fit(train_x,train_y)
#获取最优的参数
print(model.best_params_)
#将最优的参数带到岭回归模型训练
rd1 = Ridge(alpha=model.best_params_['alpha'])
#训练
rd1.fit(train_x,train_y)
#预测
y_pred = rd1.predict(test_x)


















































'''
总结，最后，会选择随机森林和梯度提升树进行比较，调参得到最好的模型进行处理
1.6.3. 字段分析
人为分析每个字段的含义，之间存在的关联，在数据分析时可以更好的体现数据之间的关联性

1.6.4. 字段处理（数据挖掘） 日期处理
本身字段没有使用价值，可以对字段进行拆分或者合并的方式，将字段拆解成更多有利于分析的信息

1.6.5. 噪音处理
噪音肯定会影响最终的判断结果，需要将训练集的噪音进行去除，依据 样本-均值和标准差之间的关系

1.6.6. 数据分析
分析每个字段和标签之间的关系，以及字段之间是否存在关联

1.6.7. 数据预处理
离散值进行onehot处理，连续值进行标准化处理

1.6.8. 应用处理
尽可能使用多种模型进行预算，同时使用网格搜索配合大量参数进行拟合，选出最优参数
'''











