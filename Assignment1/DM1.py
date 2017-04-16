#-*-coding:utf-8-*-
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# f1 = open("data/horse-colic.data", 'r')
# f2 = open("data/horse-colic.csv", 'w')
#
# line = f1.readline()
# while (line):
#     temp = line.strip().split()
#     temp = ','.join(temp) + '\n'
#     f2.write(temp)
#     line = f1.readline()
#
# f1.close()
# f2.close()

name_category = ["n1", "n2", "n3", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n17", "n18", "n21", "n23", "n24","n25","n26","n27", "n28"]
name_value = ["v4", "v5", "v6", "v16", "v19", "v20", "v22"]
nameall=["n1", "n2","n3","v4","v5","v6","n7","n8","n9","n10","n11","n12","n13","n14","n15","v16","n17","n18","v19","v20","n21","v22","n23","n24","n25","n26","n27","n28"]

data_raw = pd.read_csv("data/horse-colic.csv",names=nameall,na_values="?")

# # 对标称属性，给出每个可能取值的频数
# for item in name_category:
#     print item
#     print pd.value_counts(data_raw[item].values)

# # - 对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数
#
# # 最大值
# data_show = pd.DataFrame(data=data_raw[name_value].max(), columns=['max'])
# # 最小值
# data_show['min'] = data_raw[name_value].min()
# # 均值
# data_show['mean'] = data_raw[name_value].mean()
# # 中位数
# data_show['median'] = data_raw[name_value].median()
# # 四分位数
# data_show['quartile'] = data_raw[name_value].describe().loc['25%']
# # 缺失值个数
# data_show['missing'] = 368-data_raw[name_value].describe().loc['count']#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# print data_show


# # 直方图
#
# fig = plt.figure(figsize=(20, 11))
# i = 1
# for item in name_value:
#     ax = fig.add_subplot(3, 5, i)
#     data_raw[item].plot(kind='hist', title=item, ax=ax)
#     i += 1
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# fig.savefig('image/histogram.jpg')

# # QQ图
# fig = plt.figure(figsize=(20, 12))
# i = 1
# for item in name_value:
#     ax = fig.add_subplot(3, 5, i)
#     sm.qqplot(data_raw[item], ax=ax)
#     ax.set_title(item)
#     i += 1
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# fig.savefig('image/qqplot.jpg')

# # 盒图
# fig = plt.figure(figsize=(20, 12))
# i = 1
# for item in name_value:
#     ax = fig.add_subplot(3, 5, i)
#     data_raw[item].plot(kind='box')
#     i += 1
# fig.savefig('image/boxplot.jpg')
#
#



# #1.将缺失部分剔除
# data_fill = data_raw.dropna()
#
#
# fig = plt.figure(figsize=(25, 15))
# i = 1
# for item in nameall:
#     ax = fig.add_subplot(5, 6, i)
#     ax.set_title(item)
#     data_raw[item].plot(ax=ax, alpha=0.5, kind='hist', label='raw', legend=True)
#     data_fill[item].plot(ax=ax, alpha=0.5, kind='hist', label='fill', legend=True)
#     i += 1
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
#
#
# fig.savefig('image/fill_delete.jpg')
# data_fill.to_csv('data/fill_delete.csv', mode='w', encoding='utf-8', index=False, header=False)


# # 2.用最高频率值来填补缺失值
#
#
# data_fill = data_raw.copy()
# for item in name_category + name_value:
#     # 计算最高频率的值
#     most_frequent_value = data_fill[item].value_counts().idxmax()
#     # 替换缺失值
#     data_fill[item].fillna(value=most_frequent_value, inplace=True)
#
# fig = plt.figure(figsize=(25, 15))
# i = 1
# for item in nameall:
#     ax = fig.add_subplot(5, 6, i)
#     ax.set_title(item)
#     data_raw[item].plot(ax=ax, alpha=0.5, kind='hist', label='raw', legend=True)
#     data_fill[item].plot(ax=ax, alpha=0.5, kind='hist', label='fill', legend=True)
#     i += 1
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
#
#
# fig.savefig('image/fill_most.jpg')
# data_fill.to_csv('data/fill_most.csv', mode='w', encoding='utf-8', index=False, header=False)


# # 3. 通过属性的相关关系来填补缺失值
#
#
# data_fill = data_raw.copy()
# # 对数值型属性的每一列，进行插值运算
# for item in name_value:
#     data_fill[item].interpolate(inplace=True)
#
# fig = plt.figure(figsize=(25, 15))
# i = 1
# for item in nameall:
#     ax = fig.add_subplot(5, 6, i)
#     ax.set_title(item)
#     data_raw[item].plot(ax=ax, alpha=0.5, kind='hist', label='raw', legend=True)
#     data_fill[item].plot(ax=ax, alpha=0.5, kind='hist', label='fill', legend=True)
#     i += 1
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
#
# fig.savefig('image/fill_corelation.jpg')
# data_fill.to_csv('data/fill_corelation.csv', mode='w', encoding='utf-8', index=False,header=False)
#
#
# 4.通过数据对象之间的相似性来填补缺失值

# 将缺失值设为0，对数据集进行正则化

# 建立原始数据的拷贝，用于正则化处理
data_norm = data_raw.copy()
# 将数值属性的缺失值替换为0
data_norm[name_value] = data_norm[name_value].fillna(0)
# 对数据进行正则化
data_norm[name_value] = data_norm[name_value].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

nan_list = pd.isnull(data_raw).any(1).nonzero()[0]

# 构造分数表
score = {}
range_length = len(data_raw)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0

# 在处理后的数据中，对每两条数据条目计算差异性得分，分值越高差异性越大
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in name_category:
            if data_norm.iloc[i][item] != data_norm.iloc[j][item]:
                score[i][j] += 1
        for item in name_value:
            temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
            score[i][j] += temp
        score[j][i] = score[i][j]

# 建立原始数据的拷贝
data_fill = data_raw.copy()

# 对有缺失值的条目，用和它相似度最高（得分最低）的数据条目中对应属性的值替换
for index in nan_list:
    best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse=False)[1][0]
    for item in name_value:
        if pd.isnull(data_fill.iloc[index][item]):
            if pd.isnull(data_raw.iloc[best_friend][item]):
                data_fill.ix[index, item] = data_raw[item].value_counts().idxmax()
            else:
                data_fill.ix[index, item] = data_raw.iloc[best_friend][item]

fig = plt.figure(figsize=(25, 15))
i = 1
for item in nameall:
    ax = fig.add_subplot(5, 6, i)
    ax.set_title(item)
    data_raw[item].plot(ax=ax, alpha=0.5, kind='hist', label='raw', legend=True)
    data_fill[item].plot(ax=ax, alpha=0.5, kind='hist', label='fill', legend=True)
    i += 1
plt.subplots_adjust(wspace=0.3, hspace=0.3)



# 保存图像和处理后数据
fig.savefig('image/fill_similarity.jpg')
data_fill.to_csv('data/fill_similarity.csv', mode='w', encoding='utf-8', index=False, header=False)


