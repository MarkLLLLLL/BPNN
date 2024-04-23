# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import palettable
from plotnine import *
sns.set_style("darkgrid")
sns.set_context("paper")
# %matplotlib inline
plt.rcParams["font.sans-serif"]='SimHei'   #解决中文乱码问题
plt.rcParams['axes.unicode_minus']=False   #解决负号无法显示的问题
import warnings
warnings.filterwarnings('ignore')
# 数据读取
# data = pd.read_csv('C:/Users/lmz/Desktop/数据/data11.csv',engine = 'python',encoding='gbk')
# data.head()
# 相关性分析
# 数据读取
data = pd.read_csv('C:/Users/lmz/Desktop/数据/data10.csv',engine = 'python',encoding='utf-8')
plt.figure(figsize=(7, 6))
cor=data.corr()                 #默认为'pearson'检验，可选'kendall','spearman'
sns.heatmap(cor,
            annot = True,       # 是否显示数值
            linewidths = 0.2,   # 格子边线宽度
            vmax=0.5,           # 颜色变浅
            cbar = True,        # 是否显示图例色带
           )
plt.savefig('C:/Users/lmz/Desktop/输出图形/data11.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
cor.to_csv('C:/Users/lmz/Desktop/输出图形/data11.csv')
cor
# # 相关性分析
# plt.figure(figsize=(13, 13))
# cor=data.corr(method='spearman')                 #默认为'pearson'检验，可选'kendall','spearman'
# sns.heatmap(cor,
#             annot = True,       # 是否显示数值
#             linewidths = 0.2,   # 格子边线宽度
#             vmax=0.5,           # 颜色变浅
#             cbar = True,        # 是否显示图例色带
#            )
# plt.savefig('C:/Users/lmz/Desktop/输出图形/ze2.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# cor.to_csv('C:/Users/lmz/Desktop/输出图形/a2.csv')
# cor



