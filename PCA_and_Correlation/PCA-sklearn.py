import numpy as np
from sklearn.decomposition import PCA
from xlwt import Workbook
import xlsxwriter
import xlrd
import openpyxl
import csv
import array
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from mpl_toolkits.mplot3d import axes3d
import math
import seaborn as sns
import pandas as pd

##data = pd.read_csv("C:\\Users\\lmz\\Desktop\\test2.csv")
##data = pd.read_table("C:\\Users\\lmz\\Desktop\\test2.csv",sep=",")

# 加快读取数据的速度
df = pd.DataFrame()
chunksize = 10    #这个数字设置多少有待考察
for chunk in pd.read_csv("C:\\Users\\lmz\\Desktop\\Microstructures\\结果数据\\PCA_initial.csv", chunksize=chunksize):
# for chunk in pd.read_csv("C:\\Users\\lmz\\Desktop\\test2.csv", chunksize=chunksize):
    df = df.append(chunk)

X=df

##data = openpyxl.load_workbook('C:\\Users\\lmz\\Desktop\\Coporation_evaluation.xlsx')
##sh=data.active
### 1.获取矩形区域中的单元格对象
##cell_tuples = sh['B2': 'I16']
##X=[]
##for i in range (len(cell_tuples)):
##    Y=[]
##    for j in range (len(cell_tuples[1])):
##        Y.append(cell_tuples[i][j].value)
##    X.append(Y)

X=np.asarray(X)
pca = PCA(n_components=50)
pca.fit(X)
# 输出特征值
print('特征值')
print(pca.explained_variance_)
# 输出特征值贡献率
print('特征值贡献率')
print(pca.explained_variance_ratio_)
# 输出特征向量
print('特征向量')
print(pca.components_)
# 降维后的数据
print('降维后的数据')
X_new = pca.transform(X)
print(X_new)
##fig = plt.figure()
##plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
##plt.show()

# 2.绘制图片
len_matrix=int(math.sqrt(len(pca.components_[0])/3))
PC1_1 = np.mat(pca.components_[0][0:int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC1_2 = np.mat(pca.components_[0][int(len(pca.components_[0])/3):2*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC1_3 = np.mat(pca.components_[0][2*int(len(pca.components_[0])/3):3*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))

PC2_1 = np.mat(pca.components_[1][0:int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC2_2 = np.mat(pca.components_[1][int(len(pca.components_[0])/3):2*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC2_3 = np.mat(pca.components_[1][2*int(len(pca.components_[0])/3):3*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))

PC3_1 = np.mat(pca.components_[2][0:int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC3_2 = np.mat(pca.components_[2][int(len(pca.components_[0])/3):2*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC3_3 = np.mat(pca.components_[2][2*int(len(pca.components_[0])/3):3*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))

PC4_1 = np.mat(pca.components_[3][0:int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC4_2 = np.mat(pca.components_[3][int(len(pca.components_[0])/3):2*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))
PC4_3 = np.mat(pca.components_[3][2*int(len(pca.components_[0])/3):3*int(len(pca.components_[0])/3)].reshape(len_matrix,len_matrix))

##plt.figure("PC", facecolor="lightgray")
plt.matshow(PC1_1, cmap=plt.cm.jet, vmin=PC1_1.min(), vmax=PC1_1.max())
plt.colorbar()
plt.title('PC1_1', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1_1.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC1_2, cmap=plt.cm.jet, vmin=PC1_2.min(), vmax=PC1_2.max())
plt.colorbar()
plt.title('PC1_2', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1_2.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC1_3, cmap=plt.cm.jet, vmin=PC1_3.min(), vmax=PC1_3.max())
plt.colorbar()
plt.title('PC1_3', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1_3.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC2_1, cmap=plt.cm.jet, vmin=PC2_1.min(), vmax=PC2_1.max())
plt.colorbar()
plt.title('PC2_1', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2_1.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC2_2, cmap=plt.cm.jet, vmin=PC2_2.min(), vmax=PC2_2.max())
plt.colorbar()
plt.title('PC2_2', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2_2.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC2_3, cmap=plt.cm.jet, vmin=PC2_3.min(), vmax=PC2_3.max())
plt.colorbar()
plt.title('PC2_3', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2_3.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC3_1, cmap=plt.cm.jet, vmin=PC3_1.min(), vmax=PC3_1.max())
plt.colorbar()
plt.title('PC3_1', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3_1.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC3_2, cmap=plt.cm.jet, vmin=PC3_2.min(), vmax=PC3_2.max())
plt.colorbar()
plt.title('PC3_2', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3_2.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC3_3, cmap=plt.cm.jet, vmin=PC3_3.min(), vmax=PC3_3.max())
plt.colorbar()
plt.title('PC3_3', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3_3.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC4_1, cmap=plt.cm.jet, vmin=PC4_1.min(), vmax=PC4_1.max())
plt.colorbar()
plt.title('PC4_1', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4_1.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC4_2, cmap=plt.cm.jet, vmin=PC4_2.min(), vmax=PC4_2.max())
plt.colorbar()
plt.title('PC4_2', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4_2.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.matshow(PC4_3, cmap=plt.cm.jet, vmin=PC4_3.min(), vmax=PC4_3.max())
plt.colorbar()
plt.title('PC4_3', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4_3.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
##plt.show()
# plt.matshow(PC1, cmap=plt.cm.jet)
# plt.title('PC1', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC2, cmap=plt.cm.jet)
# plt.title('PC2', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC3, cmap=plt.cm.jet)
# plt.title('PC3', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC4, cmap=plt.cm.jet)
# plt.title('PC4', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
##plt.show()

plt.figure("3D Scatter", facecolor="lightgray")
ax3d = plt.gca(projection="3d")  # 创建三维坐标

plt.title('3D Scatter', fontsize=10)
ax3d.set_xlabel('PC1', fontsize=10)
ax3d.set_ylabel('PC2', fontsize=10)
ax3d.set_zlabel('PC3', fontsize=10)
plt.tick_params(labelsize=10)

ax3d.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], s=20, cmap="jet", marker="o")

plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/3D Scatter.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')

# plt.show()

#画出PC1和PC2的图
plt.figure(figsize=(2, 2), dpi=400)
plt.scatter(X_new[:,0], X_new[:,1])
plt.title('PC1_PC2', fontsize=10)
plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1_PC2.png',
            dpi=400,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'b')
plt.show()



##from sklearn.decomposition import PCA
##import pandas as pd
##
####X = pd.read_csv('C:\\Users\\lmz\\Desktop\\PCA_data.csv')
### Standardize
### X = X - X.mean(axis=0)
##
##pca = PCA(n_components=3)
##newX = pca.fit_transform(X)
##print(newX)


##！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1
##  适用于两相材料的一个两点相关函数降维
# import numpy as np
# from sklearn.decomposition import PCA
# from xlwt import Workbook
# import xlrd
# import openpyxl
# import csv
# import array
# import matplotlib.pyplot as plt
# from sklearn.datasets._samples_generator import make_blobs
# from mpl_toolkits.mplot3d import axes3d
# import math
# import seaborn as sns
# import pandas as pd

# ##data = pd.read_csv("C:\\Users\\lmz\\Desktop\\test2.csv")
# ##data = pd.read_table("C:\\Users\\lmz\\Desktop\\test2.csv",sep=",")

# # 加快读取数据的速度
# df = pd.DataFrame()
# chunksize = 10    #这个数字设置多少有待考察
# for chunk in pd.read_csv("C:\\Users\\lmz\\Desktop\\Microstructures\\结果数据\\PCA_initial.csv", chunksize=chunksize):
# # for chunk in pd.read_csv("C:\\Users\\lmz\\Desktop\\test2.csv", chunksize=chunksize):
#     df = df.append(chunk)

# X=df

# ##data = openpyxl.load_workbook('C:\\Users\\lmz\\Desktop\\Coporation_evaluation.xlsx')
# ##sh=data.active
# ### 1.获取矩形区域中的单元格对象
# ##cell_tuples = sh['B2': 'I16']
# ##X=[]
# ##for i in range (len(cell_tuples)):
# ##    Y=[]
# ##    for j in range (len(cell_tuples[1])):
# ##        Y.append(cell_tuples[i][j].value)
# ##    X.append(Y)

# X=np.asarray(X)
# pca = PCA(n_components=4)
# pca.fit(X)
# # 输出特征值
# print('特征值')
# print(pca.explained_variance_)
# # 输出特征值贡献率
# print('特征值贡献率')
# print(pca.explained_variance_ratio_)
# # 输出特征向量
# print('特征向量')
# print(pca.components_)
# # 降维后的数据
# print('降维后的数据')
# X_new = pca.transform(X)
# print(X_new)
# ##fig = plt.figure()
# ##plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
# ##plt.show()

# # 2.绘制图片
# len_matrix=int(math.sqrt(len(pca.components_[0])))
# PC1 = np.mat(pca.components_[0].reshape(len_matrix,len_matrix))
# PC2 = np.mat(pca.components_[1].reshape(len_matrix,len_matrix))
# PC3 = np.mat(pca.components_[2].reshape(len_matrix,len_matrix))
# PC4 = np.mat(pca.components_[3].reshape(len_matrix,len_matrix))

# ##plt.figure("PC", facecolor="lightgray")
# plt.matshow(PC1, cmap=plt.cm.hot, vmin=PC1.min(), vmax=PC1.max())
# plt.colorbar()
# plt.title('PC1', fontsize=10)
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1_1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC2, cmap=plt.cm.hot, vmin=PC2.min(), vmax=PC2.max())
# plt.colorbar()
# plt.title('PC2', fontsize=10)
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2_1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC3, cmap=plt.cm.hot, vmin=PC3.min(), vmax=PC3.max())
# plt.colorbar()
# plt.title('PC3', fontsize=10)
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3_1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC4, cmap=plt.cm.hot, vmin=PC4.min(), vmax=PC4.max())
# plt.colorbar()
# plt.title('PC4', fontsize=10)
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4_1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# ##plt.show()
# plt.matshow(PC1, cmap=plt.cm.jet)
# plt.title('PC1', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC1.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC2, cmap=plt.cm.jet)
# plt.title('PC2', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC2.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC3, cmap=plt.cm.jet)
# plt.title('PC3', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC3.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# plt.matshow(PC4, cmap=plt.cm.jet)
# plt.title('PC4', fontsize=10)
# plt.colorbar()
# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/PC4.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')
# ##plt.show()

# plt.figure("3D Scatter", facecolor="lightgray")
# ax3d = plt.gca(projection="3d")  # 创建三维坐标

# plt.title('3D Scatter', fontsize=10)
# ax3d.set_xlabel('PC1', fontsize=10)
# ax3d.set_ylabel('PC2', fontsize=10)
# ax3d.set_zlabel('PC3', fontsize=10)
# plt.tick_params(labelsize=10)

# ax3d.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], s=20, cmap="jet", marker="o")

# plt.savefig('C:/Users/lmz/Desktop/Microstructures/PC/3D Scatter.png',
#             dpi=400,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'b')

# plt.show()






# ##from sklearn.decomposition import PCA
# ##import pandas as pd
# ##
# ####X = pd.read_csv('C:\\Users\\lmz\\Desktop\\PCA_data.csv')
# ### Standardize
# ### X = X - X.mean(axis=0)
# ##
# ##pca = PCA(n_components=3)
# ##newX = pca.fit_transform(X)
# ##print(newX)



# list1 = [[0 for j in range(4)] for i in range(4)]  #创建一个二维列表
list1=X_new
output = open('PCA-test.xls','w',encoding='gbk')  #不需要事先创建一个excel表格，会自动生成，gbk为编码方式，支持中文，w代表write
output.write('1\t2\t3\t4\n')
for i in range(len(list1)):
	for j in range(len(list1[i])):
		output.write(str(list1[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()

# 存放特征向量，也就是主成分矩阵
# list2=pca.components_
# output = open('test.xls','w',encoding='gbk')  #不需要事先创建一个excel表格，会自动生成，gbk为编码方式，支持中文，w代表write
# # output.write('1\t2\t3\t4\n')
# for i in range(len(list2)):
# 	for j in range(len(list2[i])):
# 		output.write(str(list2[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
# 		output.write('\t')   #相当于Tab一下，换一个单元格
# 	output.write('\n')       #写完一行立马换行
# output.close()


# 写出csv文件用于绘制主成分向量图
np.savetxt('pca.components_.csv', pca.components_, delimiter=',')

np.savetxt('pca.explained_variance_ratio_.csv', pca.explained_variance_ratio_, delimiter=',')
