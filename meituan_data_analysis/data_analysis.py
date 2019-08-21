import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wordcloud
from scipy.cluster.hierarchy import linkage,dendrogram   # scipy的层次聚类函数
from sklearn.cluster import AgglomerativeClustering    # 导入sklearn的层次聚类函数
# 解决无法显示中文问题

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文
plt.rcParams['axes.unicode_minus'] = False   # 用于正常显示负号
import pylab as pl

# data = pd.read_csv('meituan.csv')
#
# data_5 = data[['dish_type','restaurant_name','price','star','comment_num']]
# data_5.to_csv('data_5.csv')
# data_location = data['location']
#
# data_location = data_location.str.split(u'区',expand=True)[0]
# data_location.to_csv('location.csv')
#
# data_location = pd.read_csv('location.csv',names=['location'])
# data_location = data_location['location'] + u'区'
# data_location.to_csv('location.csv')
#
# data_location = pd.read_csv('location.csv',names=['location'])
# data_location['location'] = data_location[data_location['location'].str.len()<=4]
# data_location = data_location.fillna(u'其他区')   # 根据字符串长度将其转换空值再进行赋值
# print(data_location)
# data_location.to_csv('location.csv')
#
# data_all = data_5.join(data_location)
# data_all.to_csv('meituan_cleaned.csv')
# print(data_all)


# '''根据各个区的店铺数量画出条形图'''
# data = pd.read_csv('meituan_cleaned.csv')
# data_location = data['location']
# result_location = Counter(data_location)    # 统计各个区的店铺数量
# print(result_location)
# index_location = list(result_location.keys())
# index_location.sort(reverse=True)
# print(index_location)
# count_location = list(result_location.values())
# count_location.sort(reverse=True)
# print(count_location)
# plt.bar(index_location,count_location,width=0.5,color='b')
# plt.title('location')
# plt.xticks(rotation=-45)
# pic_output = 'location'
# plt.savefig(u'%s.png' %(pic_output))
#
# '''各地区商家和各种美食热力图'''
# data = pd.read_csv('meituan_cleaned.csv')
# data= data[['location','dish_type']]
# print(data)
# data_l_d = data.groupby(['location','dish_type']).size()
# print(data_l_d)
# data_l_d.to_csv('data_l_d.csv')
# data_l_d = dict(data_l_d)
# print(data_l_d)
# data_l_d_location = data_l_d.keys()
# print(data_l_d_location)




'''根据美食分类画出条形图'''
# data = pd.read_csv('meituan_cleaned.csv')
# data_dishtype = data['dish_type']
# result_dishtype = Counter(data_dishtype)   # 统计全部美食分类
# print(result_dishtype)
#
# index_dishtype = list(result_dishtype.keys())
# index_dishtype.sort(reverse=True)
# count_dishtype = list(result_dishtype.values())
# count_dishtype.sort(reverse=True)
# plt.bar(index_dishtype,count_dishtype,width=0.5,color='b')
# pl.xticks(rotation=-70)    # 设置底部标签旋转
# plt.title('dish_type')
# pic_output = 'dish_type'
# plt.savefig(u'%s.png' %(pic_output),dpi=500,bbox_inches='tight')
# dpi=500是设置保存图片的像素高低，plt.savefig里面的参数 bbox_inches='tight'保证保存图片完整性


# '''根据美食分类画饼状图'''
# data = pd.read_csv('meituan_cleaned.csv')
# print(data)
# data_dishtype = data['dish_type']
# data_dishtype = Counter(data_dishtype)
# labels = list(data_dishtype.keys())
# print(labels)
# count= list(data_dishtype.values())
# print(count)
#
'''饼状图'''
# plt.axes(aspect=1)  # 如果说饼状图不是平面圆，可以用此方法
# plt.pie(x=count, labels=labels, autopct='%1.1f%%', shadow=True)
# plt.title('dish_type')

'''词云展示'''
# mask = np.array(Image.open('wordclound.jpg'))
# wc = wordcloud.WordCloud(
#     font_path='C://Windows//Fonts//simhei.ttf',
#     mask=mask,
#     max_words=27,
#     max_font_size=100
# )
# wc.generate_from_frequencies(result_dishtype)
# image_colors = wordcloud.ImageColorGenerator(mask)
# wc.recolor(color_func=image_colors)
# pic_output = 'wordclound'
# plt.imshow(wc)
#
# plt.axis('off')  # 关闭坐标轴
# plt.savefig(u'%s.png' %(pic_output))

'''人均消费和评分分组求平均值'''
# data = pd.read_csv('meituan_cleaned.csv')
# data_dsp = data[['star','dish_type','price']]
# # print(data_dsp)
# data_typedsp = data_dsp.groupby(['dish_type'])[['star','price']].mean()   # 用groupby函数进行分组计算
# print(data_typedsp)
#
# dish_type =data_typedsp.index
# dish_type = list(dish_type)
# star_mean = data_typedsp[data_typedsp.columns[0]]
# star_mean = list(star_mean)
# price_mean = data_typedsp[data_typedsp.columns[1]]
# price_mean = list(price_mean)
# print(dish_type)
# print(star_mean)
# print(price_mean)

'''人均消费和评分平均评分画出折线形图'''
# line1,=plt.plot(dish_type,star_mean,linestyle='-',color='b',marker='o',label='avgstar')
# pl.xticks(rotation=-90)
#
# plt.twinx()   # 增加一个y轴
# line2,=plt.plot(dish_type,price_mean,linestyle='--',color='g',marker='v',label='avgprice')
# pl.xticks(rotation=-90)
# plt.title('avgstar and avgprice')

'''打开图例（标识分类线）'''
# plt.legend(loc=0, ncol=2)
# plt.legend([line1,line2], ['avgstar','avgprice'])
# pic_output = 'Avg_price_star_plot'
# plt.savefig(u'%s.png' %(pic_output), dpi=500,bbox_inches='tight')

'''散点图'''
# plt.scatter(star_mean,price_mean,s=500,color='r',marker='o',alpha=0.5)
# plt.title('Avg_price_star_scatter')
# pic_output = 'ps_scatter'
# plt.savefig(u'%s.png' %(pic_output),dpi=500)




'''评分和评价数量的平均值'''
# data = pd.read_csv('meituan_cleaned.csv')
# data_dsc = data[['dish_type','star','comment_num']]
# print(data_dsc)
#
# data_typeds = data_dsc.groupby(['dish_type'])['star'].mean()
# print(data_typeds)
# data_typedc = data_dsc.groupby(['dish_type'])['comment_num'].sum()
# print(data_typedc)
# data_typedsc = pd.merge(data_typeds,data_typedc,how='left',on='dish_type')
# print(data_typedsc)
#
# dish_type = data_typedsc.index
# dish_type = list(dish_type)
# print(dish_type)
# star_mean = data_typedsc[data_typedsc.columns[0]]
# star_mean = list(star_mean)
# print(star_mean)
# comment_num = data_typedsc[data_typedsc.columns[1]]
# comment_num = list(comment_num)
# print(comment_num)
#
# plt.ylim([0,5])    # 重新调整坐标轴刻度
#
# line1,=plt.plot(dish_type,star_mean,linestyle='--',color='g',marker='o',label='avgstar')
# plt.xticks(rotation=-90)
#
# plt.twinx()
# plt.bar(dish_type,comment_num,width=0.5,color='b')
#
# # plt.xticks(rotation=-90)
# plt.legend(loc=0,ncol=1)
# plt.legend([line1],['avgstar','avcomment'])
# plt.title('Avgstar and Comment_num')
#
# pic_output = 'Avgstar_commentnum'
# plt.savefig(u'%s.png' %(pic_output), dpi=500, bbox_inches='tight')

'''散点图'''
# plt.scatter(star_mean,comment_num,s=500,color='r',marker='o',alpha=0.5)
# plt.title('scatter')





'''商家商铺圈分析'''
# 数据标准化到[0,1]

data = pd.read_csv('meituan_cleaned.csv',index_col='restaurant_name')
data = data[['star','price','comment_num']]

data = (data-data.min())/(data.max()-data.min())
# data = data.reset_index()   # 重新设立索引

data.to_csv('meituan_standard.csv')

'''模型构建（商圈聚类，画出谱系聚类图）'''
data = pd.read_csv('meituan_standard.csv',index_col='restaurant_name')
print(data)

Z = linkage(data, method='ward', metric='euclidean')
P = dendrogram(Z, 0)
pic_output = 'cluster'
plt.title(u'谱系聚类图')
plt.savefig(u'%s.png' %(pic_output),dpi=500)

'''层次聚类算法'''

k = 3  # 从谱系聚类图中看出有几类

data = pd.read_csv('meituan_standard.csv', index_col='restaurant_name')
# print(data)

model = AgglomerativeClustering(n_clusters=k, linkage='ward')
model.fit(data)   # 模型训练

# 详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
r.columns = list(data.columns) + [u'聚类类别']
print(r)


style = ['ro-','go-','bo-']
title = [u'商铺圈类别1',u'商铺圈类别2',u'商铺圈类别3']
xlabels = ['star','price','comment_num']
pic_output = 'type_'


for i in range(k):
    plt.figure()
    tmp = r[r[u'聚类类别']==i].iloc[:,:3]
    for j in range(len(tmp)):
        plt.plot(range(1,4), tmp.iloc[j], style[i])

    plt.xticks(range(1,4), xlabels, rotation=20)  # 坐标标签
    plt.title(title[i])
    plt.subplots_adjust(bottom=0.15)  # 调整底部
    plt.savefig(u'%s%s.png' %(pic_output,i),dpi=500)   # 保存图片

plt.tight_layout() # 使图片能够完整显示，自动调整参数
plt.style.use('ggplot')  # 进行格式美化
plt.show()





