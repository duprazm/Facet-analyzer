 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:26:41 2021

@author: dupraz
"""

import numpy as np
from scipy.linalg import norm
import matplotlib
#import csv
#from mayavi import mlab
#import matplotlib
import math
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
#import os
#import vtk

filename = 'Delta_disp.txt'

pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

savedir = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

data = np.loadtxt(pathload+filename)

# Relaxation displacement last disp vs average, average facets


hkl_avg = ['ZX','ZX','-YZ','-YZ','YZ','YZ','-ZX','-ZX']

hkl_zx =[[-1,0,0],[-1,0,0],[-1,1,1],[-1,1,1],[-1,1,3],[-1,1,3],
         [-3,1,1],[-3,1,1],[-1,0,1],[-1,0,1]]
hkl_myz=[[0,1,0],[0,1,0],[-1,1,-1],[-1,1,-1],[-1,1,0],[-1,1,0],
         [0,1,-1],[0,1,-1]]
hkl_yz=[[0,-1,0],[0,-1,0],[1,-1,1],[1,-1,1],[-1,-1,1],[-1,-1,1],
        [0,-1,1],[0,-1,1],[-1,-1,3],[-1,-1,3],[1,-3,1],[1,-3,1]]
hkl_mzx=[[1,0,0],[1,0,0],[0,0,-1],[0,0,-1],[1,-1,-1],[1,-1,-1],
        [1,0,-1],[1,0,-1],[1,-1,3],[1,-1,3],[1,1,-3],[1,1,-3]]


plt.figure(figsize=(15,10))

plt.plot(data[5,20:-10:5],'-xr')
plt.plot(data[5,22:-10:5],'-or')
plt.plot(data[10,20:-10:5],'-xb')
plt.plot(data[10,22:-10:5],'-ob')
plt.plot(data[17,20:-10:5],'-xg')
plt.plot(data[17,22:-10:5],'-og')
plt.plot(data[24,20:-10:5],'-xm')
plt.plot(data[24,22:-10:5],'-om')
plt.legend(hkl_avg, fancybox=True, shadow=True)
plt.savefig(savedir+'Average_delta_disp_ref_02_4_group.png')
plt.show()

# Relaxation displacement +ZX



plt.figure(figsize=(15,10))
plt.plot(data[0,20:-10:5],'-xr')
plt.plot(data[0,22:-10:5],'-or')
plt.plot(data[1,20:-10:5],'-xb')
plt.plot(data[1,22:-10:5],'-ob')
plt.plot(data[2,20:-10:5],'-xg')
plt.plot(data[2,22:-10:5],'-og')
plt.plot(data[3,20:-10:5],'-xm')
plt.plot(data[3,22:-10:5],'-om')
plt.plot(data[4,20:-10:5],'-xc')
plt.plot(data[4,22:-10:5],'-oc')
plt.legend(hkl_zx,fancybox=True, shadow=True)
plt.savefig(savedir+'Average_delta_disp_ref_02_4_ZX_group.png')
plt.show()

# Relaxation displacement -YZ

plt.figure(figsize=(15,10))
plt.plot(data[6,20:-10:5],'-xr')
plt.plot(data[6,22:-10:5],'-or')
plt.plot(data[7,20:-10:5],'-xb')
plt.plot(data[7,22:-10:5],'-ob')
plt.plot(data[8,20:-10:5],'-xg')
plt.plot(data[8,22:-10:5],'-og')
plt.plot(data[9,20:-10:5],'-xm')
plt.plot(data[9,22:-10:5],'-om')
plt.legend(hkl_myz,fancybox=True, shadow=True)
plt.savefig(savedir+'Average_delta_disp_ref_02_4_mYZ_group.png')
plt.show()

# Relaxation displacement -YZ

plt.figure(figsize=(15,10))
plt.plot(data[11,20:-10:5],'-xr')
plt.plot(data[11,22:-10:5],'-or')
plt.plot(data[12,20:-10:5],'-xb')
plt.plot(data[12,22:-10:5],'-ob')
plt.plot(data[13,20:-10:5],'-xg')
plt.plot(data[13,22:-10:5],'-og')
plt.plot(data[14,20:-10:5],'-xm')
plt.plot(data[14,22:-10:5],'-om')
plt.plot(range(1,4),data[15,25:-10:5],'-xc')
plt.plot(range(1,4),data[15,27:-10:5],'-oc')
plt.plot(range(1,4),data[16,25:-10:5],'-xk')
plt.plot(range(1,4),data[16,27:-10:5],'-ok')
plt.legend(hkl_yz,fancybox=True, shadow=True)
plt.savefig(savedir+'Average_delta_disp_ref_02_4_YZ_group.png')
plt.show()

# Relaxation displacement -ZX

plt.figure(figsize=(15,10))
plt.plot(data[18,20:-10:5],'-xr')
plt.plot(data[18,22:-10:5],'-or')
plt.plot(data[19,20:-10:5],'-xb')
plt.plot(data[19,22:-10:5],'-ob')
plt.plot(data[20,20:-10:5],'-xg')
plt.plot(data[20,22:-10:5],'-og')
plt.plot(data[21,20:-10:5],'-xm')
plt.plot(data[21,22:-10:5],'-om')
plt.plot(data[22,20:-10:5],'-xc')
plt.plot(data[22,22:-10:5],'-oc')
plt.plot(data[23,20:-10:5],'-xk')
plt.plot(data[23,22:-10:5],'-ok')
plt.legend(hkl_mzx,fancybox=True, shadow=True)
plt.savefig(savedir+'Average_delta_disp_ref_02_4_mZX_group.png')

plt.show()


