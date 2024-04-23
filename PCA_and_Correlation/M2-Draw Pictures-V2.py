# -*- coding:utf-8 -*-
import xlrd
import matplotlib.pyplot as plt
#调节字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc('font',family='Times New Roman')
# 导入excel文件，以及第几张表
# xls_path = r'C:\Users\lmz\Desktop\NEW2-EDP TEST\1 Ten Stress Strain curve data.xls'
for i in range(46,58):
    data1 = xlrd.open_workbook('C:\\Users\\lmz\\Desktop\\M2_convergence_study\\%d Ten Stress Strain curve data.xls'%(i))
    table1 = data1.sheets()[0]
    data2 = xlrd.open_workbook('C:\\Users\\lmz\\Desktop\\M2_convergence_study\\%d Com Stress Strain curve data.xls'%(i))
    table2 = data2.sheets()[0]
    #拉伸图的数据
    t1 = table1.col_values(0)
    # t1=t1*1000000
    tt1 = t1[1:len(t1)]
    t2 = table1.col_values(1)
    # t2=t2*1000000
    tt2 = t2[1:len(t2)]
    Ten_strength=t2[0]/1000000
    Ten_modulus=table1.col_values(3)[0]/1000
    #压缩图的数据
    t3 = table2.col_values(0)
    # t3=t3*1000000
    tt3 = t3[1:len(t3)]
    t4 = table2.col_values(1)
    # t4=t4*1000000
    tt4 = t4[1:len(t4)]
    Com_strength=t4[0]/1000000
    Com_modulus=table2.col_values(3)[0]/1000
    #画出拉伸和压缩的图
    plt.figure(figsize=(6, 6))
    plt.plot(tt1, tt2, label='Ten')
    x1=[0,0.0025]
    y1=[0,Com_modulus*0.0025]
    plt.plot(x1, y1, color='red', linewidth='2.0', linestyle='-')
    x2=[0,0.015]
    y2=[Ten_strength,Ten_strength]
    plt.plot(x2, y2, color='seagreen', linewidth='2.0', linestyle='--')
    # plt.plot(tt3, tt4, label='Com')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title("%d Ten Stress Strain curve"%(i))
    # plt.legend()
    # plt.show()
    plt.savefig('C:/Users/lmz/Desktop/M2_convergence_study/SS图片/%d Ten Stress Strain curve.png'%(i),
                dpi=400,
                bbox_inches = 'tight',
                facecolor = 'w',
                edgecolor = 'b')
    
    plt.figure(figsize=(6, 6))
    # plt.plot(tt1, tt2, label='Ten')
    plt.plot(tt3, tt4, label='Com')
    x1=[0,0.02]
    y1=[0,Com_modulus*0.02]
    plt.plot(x1, y1, color='red', linewidth='2.0', linestyle='-')
    x2=[0,0.06]
    y2=[Com_strength,Com_strength]
    plt.plot(x2, y2, color='seagreen', linewidth='2.0', linestyle='--')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title("%d Com Stress Strain curve"%(i))
    # plt.legend()
    # plt.show()
    plt.savefig('C:/Users/lmz/Desktop/M2_convergence_study/SS图片/%d Com Stress Strain curve.png'%(i),
                dpi=400,
                bbox_inches = 'tight',
                facecolor = 'w',
                edgecolor = 'b')
    plt.close()





