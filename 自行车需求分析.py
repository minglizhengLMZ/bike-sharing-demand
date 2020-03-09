# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:16:37 2020

@author: lenovo
"""

import pandas as pd
'''-------------------1.导入数据，并进行解码---------------------------------------'''
df = pd.read_csv(r'E:\python_learn\共享单车需求分析\bike-sharing-demand\train.csv',encoding='utf-8')
pd.set_option('display.max_row',4)   #设置显示的最大行数
df

'''------------------2.查看数据的整体信息-------------'''
df.info()
#2.1查看数据的变量信息
df.index #查看样本量
df.columns  #查看列名
#对数据进行描述性统计
pd.set_option('display.max_columns', None)  #设置显示全部的结果
des=df.describe()
des
#des.to_csv(r'E:\python_learn\Python_work\data\bike-sharing-demand\描述性统计.csv',encoding='utf-8', index=True)#数据导出到tsetcsv.csv#

'''----------------3.数据预处理---------------------'''
#3.1数据缺失值处理（此处无缺失值）
df.isnull().any(axis=1).sum()     #查看有缺失值的行数
df.isnull().any(axis=1).sum()/df.shape[0]     #查看有缺失值的行数所占比例
#df=df.dropna()           #返回不包含缺失值的行
#df=df.fillna(value={'gender':df['gender'].mode()[0],'age':df['age'].mean()},inplace=True)
df.shape               #数据行列数
df.dtypes              #查看数据类型     

#df['G'].fillna(5, inplace=True)    # 使用指定值填充缺失值
# df.iat[0, 6] = 3         # 修改指定位置元素值，该列其他元素为缺失值NaN
#3.2查看数据重复值
print('数据行列数:', df.shape)        # 返回数据行列数（样本量，变量数）
print('数据去重:', df.drop_duplicates().shape)        # 返回新数组，删除重复行后的行列数，说明数据不存在重复行
df.duplicated().any()            #查看是否有重复观测
#如果有重复观测则
df.drop_duplicates()
#结果显示，数据没有重复数据
#3.3 查看异常值   （超过均值2倍标准差范围被视为异常值，超过3被标准差视为高度异常）
#通过箱线图查看异常值
import seaborn as sns
import matplotlib.pyplot as plt    #可视化库
fig,axes =plt.subplots(nrows=2,ncols=2,figsize=(12,6))    #绘制2行2列的图形
sns.boxplot(x='windspeed',data=df,ax=axes[0][0])   #再一行一列处展示自变量为风速的箱线图
sns.boxplot(x='casual',data=df,ax=axes[0][1])
sns.boxplot(x='registered',data=df,ax=axes[1][0])
sns.boxplot(x='count',data=df,ax=axes[1][1])
plt.show()


'''---------------------4.数据加工 -------------------------------------'''
#转换"时间和日期"的格式, 并提取出小时,周, 日(年月日), 月（年月）, 年.
#转换格式, 并提取出小时, 星期几, 月份

df['datetime'] = pd.to_datetime(df['datetime'])  #将数据datetime转换成pands的datetime格式
df['hour'] = df.datetime.dt.hour                  #提取小时,并添加到原数组中
df['week'] = df.datetime.dt.dayofweek+1             #提取周，并添加到原数组中
df['month'] = df.datetime.dt.month                #提取月，并添加到原数组中
df['year_month'] = df.datetime.dt.strftime('%Y-%m')   #提取年和月，并添加到原数组中
df['date'] = df.datetime.dt.date
#删除datetime
df.drop('datetime', axis = 1, inplace = True)   #axis=1表明删除指定的列
df.columns
df['holiday']



'''-------------------5.特征分析-----------------------------'''
#5.1 日期和总租赁数量
import matplotlib
#设置中文字体
font = {'family': 'SimHei'}
matplotlib.rc('font', **font)
#分别计算日期和月份中位数
group_date = df.groupby('date')['count'].median()           #计算各日期租赁量的中位数
group_month = df.groupby('year_month')['count'].median()    #按照月份分组，计算各月份count的中位数
group_month.index = pd.to_datetime(group_month.index)       #把各月份作为index取值
plt.figure(figsize=(16,5))                                   #绘制折线图
plt.plot(group_date.index, group_date.values, '-', color = 'b', label = '每天租赁数量中位数', alpha=0.8)
#横轴index，数据，折现形状，折线颜色，标签，
plt.plot(group_month.index, group_month.values, '-o', color='orange', label = '每月租赁数量中位数')
plt.legend()
plt.show()

#5.2各月份和总租赁数量（箱线图）
import seaborn as sns      #绘制箱线图的包
import matplotlib
plt.figure(figsize=(10, 4))
sns.boxplot(x='month', y='count', data=df)   #以各月份为横坐标，以总租赁量的纵坐绘制箱线图
plt.show()

#5.3各周几在不同类型用户的箱线图
fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(12,8))    #绘制3行一列
sns.boxplot(x='week',y='casual',data=df,ax=axes[0])
sns.boxplot(x='week',y='registered',data=df,ax=axes[1])
sns.boxplot(x='week',y='count',data=df,ax=axes[2])
plt.show()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
sns.boxplot(x="week",y='casual' ,data=df,ax=axes[0])
sns.boxplot(x='week',y='registered', data=df, ax=axes[1])
sns.boxplot(x='week',y='count', data=df, ax=axes[2])
plt.show()


#5.4小时和总租赁数量的关系（反映趋势变化信息）
#绘制第一个子图，（节假日的租赁量小时变化数据折线图）
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False        #设置后可正常显示中文标题标签

plt.figure(1, figsize=(14, 8))
plt.subplot(221)
hour_casual = df[df.holiday==1].groupby('hour')['casual'].median()
hour_registered = df[df.holiday==1].groupby('hour')['registered'].median()
hour_count = df[df.holiday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.xticks(hour_casual.index)
plt.title('未注册用户和注册用户在节假日自行车租赁情况')
#绘制第二个子图，（工作日的租赁量小时变化数据）
plt.subplot(222)
hour_casual = df[df.workingday==1].groupby('hour')['casual'].median()
hour_registered = df[df.workingday==1].groupby('hour')['registered'].median()
hour_count = df[df.workingday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户在工作日自行车租赁情况')
plt.xticks(hour_casual.index)
#绘制第三个子图（工作日和节假日总的租赁量小时变化情况）
plt.subplot(212)
hour_casual = df.groupby('hour')['casual'].median()
hour_registered = df.groupby('hour')['registered'].median()
hour_count = df.groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户自行车租赁情况')
plt.xticks(hour_casual.index)
plt.show()



#5.5利用多变量图绘制其他变量与总租赁的关系图（多个变量之间两两绘制散点图）
sns.pairplot(df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])



'''-----------6 相关矩阵 ----------------'''
#由于多个变量不满足正态分布, 对其进行对数变换.
#6.1对数转换，两两变量绘制散点图，分析变量间的相关关系
import numpy as np
df['windspeed'] = np.log(df['windspeed'].apply(lambda x: x+1))   #对windspeed的变量值+1后进行对数变换
df['casual'] = np.log(df['casual'].apply(lambda x: x+1))         
df['registered'] = np.log(df['registered'].apply(lambda x: x+1))
df['count'] = np.log(df['count'].apply(lambda x: x+1))
sns.pairplot(df[['windspeed', 'casual', 'registered', 'count']])


#6.2绘制热力图,查看变量之间的相关程度
correlation=df.corr(method='spearman')    #计算斯皮尔曼等级相关系数
plt.figure(figsize=(12,8))
sns.heatmap(correlation,linewidths=0.2,vmax=1,vmin=-1,linecolor='w',annot=True,
            annot_kws={'size':8},square=True)


'''--------------------7 建立回归模型/////岭回归-------------------------'''
#7.1划分数据集，划分自变量X数据集，和因变量Y数据集

df.drop(['year_month','date'],axis=1,inplace=True)      #删掉月和日行
from sklearn.model_selection import train_test_split    
#由于所有用户的租赁数量是由未注册用户和注册用户相加而成, 故删除.
df.drop(['casual','registered'], axis=1, inplace=True)
x = df.drop(['count'], axis=1)
y = df['count']
#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#7.2模型训练
from sklearn.linear_model import Ridge
#这里的alpha指的是正则化项参数, 初始先设置为1.
rd = Ridge(alpha=1)       #设定岭回归的参数alpha值
rd.fit(x_train, y_train)   #使用岭回归进行拟合
print(rd.coef_)            #回归的斜率项
print(rd.intercept_)        #回归的截距项


#7.3设置参数以及训练模型
alphas = 10**np.linspace(-5, 10, 500)    #（-5-10，取500个等差数组）其取值作为10的指数，构造alphas的值
betas = []                               #创建列表存储每次的训练的模型系数
for alpha in alphas:                   #分别用500个alpha选训练模型
    rd = Ridge(alpha = alpha)
    rd.fit(x_train, y_train)
    betas.append(rd.coef_)
#绘制岭迹图
plt.figure(figsize=(8,6))
plt.plot(alphas, betas)       #绘制不同alpha的岭迹
plt.xscale('log')    #采用对数坐标轴
#添加网格线
plt.grid(True)
#坐标轴适应数据量
plt.axis('tight')
plt.title(r'正则化项参数$\alpha$和回归系数$\beta$岭迹图')  #标题
plt.xlabel(r'$\alpha$')     #x轴
plt.ylabel(r'$\beta$')      #y轴
plt.show()


#通过图像可以看出, 当alpha为10^7时所有变量岭迹趋于稳定.按照岭迹法应当取alpha=10^7. 
#由于是通过肉眼观察的, 其不一定是最佳, 采用另外一种方式: 交叉验证的岭回归.

from sklearn.linear_model import RidgeCV
from sklearn import metrics
rd_cv = RidgeCV(alphas=alphas, cv=10, scoring='r2')   #确定价差验证岭回归的参数，和折叠次数
rd_cv.fit(x_train, y_train)
rd_cv.alpha_


 #最后选出的最佳正则化项参数为1847.281437099636, 然后用这个参数进行模型训练
rd = Ridge(alpha=1847) #, fit_intercept=False
rd.fit(x_train, y_train)
print(rd.coef_)
print(rd.intercept_)


'''---------------------8.模型预测---------------------------------'''

from sklearn import metrics
from math import sqrt
#分别预测训练数据和测试数据
y_train_pred = rd.predict(x_train)
y_test_pred = rd.predict(x_test)
#分别计算其均方根误差和拟合优度
 
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = rd.score(x_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = rd.score(x_test, y_test)
print('训练集RMSE: {0}, R方: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, R方: {1}'.format(y_test_rmse, y_test_score))



'''========9.Lasso回归========'''
import numpy as np 
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取

#model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(x_train, y_train)   # 线性回归建模
print('系数矩阵:\n',model.coef_,model.intercept_)

print('线性回归模型:\n',model)
print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效

# 使用模型预测
#分别预测训练数据和测试数据
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
#分别计算其均方根误差和拟合优度
 
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = model.score(x_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = model.score(x_test, y_test)
print('训练集RMSE: {0}, R方: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, R方: {1}'.format(y_test_rmse, y_test_score))







