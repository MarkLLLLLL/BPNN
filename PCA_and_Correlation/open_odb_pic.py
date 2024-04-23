# -*- coding:utf-8 -*-
from abaqus import *
from abaqusConstants import *
import regionToolset
import os, glob
import math
import time
import os
import sys
import ctypes
import multiprocessing
import random
import numpy as np
import operator
import sys, os.path
start = time.clock()
from xlwt import Workbook
import xlrd
from openpyxl import load_workbook
import displayGroupMdbToolset as dgm
for elastic_caluation_number in range(1,100):
    session.viewports['Viewport: 1'].view.setValues(nearPlane=245.933, 
        farPlane=385.783, width=148.735, height=74.94, viewOffsetX=0.139812, 
        viewOffsetY=-0.265089)
    o1 = session.openOdb(name='E:/Temp/NEW2-EDP-2/%dTen-E11.odb'%(elastic_caluation_number))
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    #: Model: E:/Temp/NEW2-EDP-2/1Ten-E11.odb
    #: Number of Assemblies:         1
    #: Number of Assembly instances: 0
    #: Number of Part instances:     1
    #: Number of Meshes:             2
    #: Number of Element Sets:       159
    #: Number of Node Sets:          879
    #: Number of Steps:              1
    session.viewports['Viewport: 1'].enableMultipleColors()
    session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
    cmap=session.viewports['Viewport: 1'].colorMappings['Material']
    session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
    session.viewports['Viewport: 1'].disableMultipleColors()
    session.viewports['Viewport: 1'].view.fitView()
    session.graphicsOptions.setValues(backgroundStyle=SOLID, 
        backgroundColor='#FFFFFF')
    session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
        visibleEdges=NONE)
    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF, 
        state=OFF, annotations=OFF)
    session.viewports['Viewport: 1'].view.fitView()
    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(
        triadPosition=(0, 0))
    # 保存图片######################################
    # mdb.models['Model-1'].parts['Part-1'].setValues(geometryRefinement=FINE)
    # session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    #     backgroundColor='#FFFFFF')
    # session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    #     geometryEdgesInShaded=OFF)
    # session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(
    #     triadPosition=(0, 0))
    # session.viewports['Viewport: 1'].enableMultipleColors()
    # session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
    # cmap=session.viewports['Viewport: 1'].colorMappings['Material']
    # session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
    # session.viewports['Viewport: 1'].disableMultipleColors()
    # session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
    # session.pngOptions.setValues(imageSize=(4096, 3280))
    # session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    # session.printToFile(fileName='C:/Users/lmz/Desktop/Microstructures2/1-%d.png'%(elastic_caluation_number), 
    #     format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
    session.pngOptions.setValues(imageSize=(4096, 2060))
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    session.printToFile(fileName='C:/Users/lmz/Desktop/Microstructures2/1-%d.png'%(elastic_caluation_number), 
        format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
    


