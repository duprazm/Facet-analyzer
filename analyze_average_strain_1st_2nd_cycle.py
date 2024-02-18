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

filename = 'Delta_eps.txt'

#

pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

data = np.loadtxt(pathload+filename)
savedir = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

# Relaxation displacement last disp vs average, average facets




hkl_avg = ['ZX','ZX','-YZ','-YZ','YZ','YZ','-ZX','-ZX']

hkl_avg_s = ['ZX','-YZ','YZ','-ZX']

hkl_zx =[[-1,0,0],[-1,0,0],[-1,1,1],[-1,1,1],[-1,1,3],[-1,1,3],
         [-3,1,1],[-1,0,1],[-1,0,1]]
hkl_zx_s =[[-1,0,0],[-1,1,1],[-1,1,3],
         [-3,1,1],[-1,0,1]]

hkl_zx_s =['[-1 0 0]','[-1 1 1]','[-1 1 3]',
         '[-3 1 1]','[-1 0 1]']

hkl_myz=[[0,1,0],[0,1,0],[-1,1,-1],[-1,1,-1],[-1,1,0],[-1,1,0],
         [0,1,-1],[0,1,-1]]
hkl_myz_s=[[0,1,0],[-1,1,-1],[-1,1,0],
         [0,1,-1]]

hkl_yz=[[0,-1,0],[0,-1,0],[1,-1,1],[1,-1,1],[-1,-1,1],[-1,-1,1],
        [0,-1,1],[0,-1,1]]#,[-1,-1,3],[-1,-1,3],[1,-3,1],[1,-3,1]]

hkl_yz_s=[[0,-1,0],[1,-1,1],[-1,-1,1],
        [0,-1,1],[-1,-1,3],[1,-3,1]]

hkl_mzx=[[1,0,0],[1,0,0],[0,0,-1],[0,0,-1],[1,-1,-1],[1,-1,-1],
        [1,0,-1],[1,0,-1],[1,-1,3],[1,-1,3],[1,1,-3],[1,1,-3]]

hkl_mzx_s=[[1,0,0],[0,0,-1],[1,-1,-1],
        [1,0,-1],[1,-1,3],[1,1,-3]]

hkl_001 = ['[0,-1,0]','[0,0,1]','[1,0,0]','[-1,0,0]','[0,0,-1]','[0,1,0]','average']

hkl_111 = ['[1,-1,1]','[-1,-1,1]','[1,-1,-1]','[-1,1,1]','[1,1,-1]','[-1,1,-1]','average']
#[[1,-1,1],[-1,-1,1],[1,-1,-1],[1,1,1],[-1,-1,-1],[-1,1,1],[1,1,-1],[-1,1,-1]]

hkl_113 = ['[1,-3,1]','[-1,-3,1]','[1,-3,-1]','[-1,-1,3]','[-3,-1,1]','[-1,1,3]'
           ,'[1,-1,-3]','[-3,1,1]','[1,1,-3]','Average','{1 1 -3}','{1 -1 -3}']

hkl_011 = ['[0,-1,1]','[1,-1,0]','[1,0,1]','[-1,0,1]','[1,0,-1]','[0,1,1]'
           ,'[1,1,0]','[-1,1,0]','[0,1,-1]','Average','{1 -1 0}','{1,1,0}']

hkl_average = ['{1,0,0}','{1,1,1}','{1,1,3}','{1,-1,-3}','{1,-1,0}','{1,1,0}']

hkl_comb = ['{1,0,0}','{1,0,0}','{1,0,0}','{1,1,1}','{1,1,1}','{1,1,1}','{1,1,3}'
            ,'{1,1,3}','{1,1,3}','{1,-1,-3}','{1,-1,-3}','{1,-1,-3}','{1,-1,0}',
            '{1,-1,0}','{1,-1,0}','{1,1,0}','{1,1,0}','{1,1,0}']



# Average strain per facet group


plt.figure(figsize=(15,10))

plt.plot(data[5,(0,4,1,5)],'-xr')
#plt.plot(data[5,(2,6,3,7)],'-or')
plt.plot(data[10,(0,4,1,5)],'-xb')
#plt.plot(data[10,(2,6,3,7)],'-ob')
plt.plot(data[17,(0,4,1,5)],'-xg')
#plt.plot(data[17,(2,6,3,7)],'-og')
plt.plot(data[24,(0,4,1,5)],'-xm')
#plt.plot(data[24,(2,6,3,7)],'-om')
plt.legend(hkl_avg, fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_group_last_first.png')
plt.show()


#Relaxation strain +ZX

plt.figure(figsize=(15,10))

plt.plot(data[0,(0,4,1,5)],'-xr')
#plt.plot(data[0,(2,6,3,7)],'-or')
plt.plot(data[1,(0,4,1,5)],'-xb')
#plt.plot(data[1,(2,6,3,7)],'-ob')
#plt.plot((0,2,3),data[2,(0,1,5)],'-xg')
plt.plot(data[2,(2,6,3,7)],'-og')
#plt.plot(data[3,(0,4,1,5)],'-xm')
plt.plot(2,data[3,3],'-om')
#plt.plot(data[4,(0,4,1,5)],'-xc')
plt.plot(data[4,(2,6,3,7)],'-oc')
plt.legend(hkl_zx_s, fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_ZX_last_first.png')
plt.show()



# # Relaxation strain -YZ

plt.figure(figsize=(15,10))

plt.plot(data[6,(0,4,1,5)],'-xr')
#plt.plot(data[6,(2,6,3,7)],'-or')
plt.plot(data[7,(0,4,1,5)],'-xb')
#plt.plot(data[7,(2,6,3,7)],'-ob')
plt.plot(data[8,(0,4,1,5)],'-xg')
#plt.plot(data[8,(2,6,3,7)],'-og')
plt.plot(data[9,(0,4,1,5)],'-xm')
#plt.plot(data[9,(2,6,3,7)],'-om')
plt.legend(hkl_myz_s,fancybox=True, shadow=True,loc='lower right',prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_mYZ_last_first.png')
plt.show()


# # Relaxation strain +YZ
plt.figure(figsize=(15,10))

#plt.plot((0,2,3),data[11,(0,1,5)],'-xr')
plt.plot(data[11,(2,6,3,7)],'-or')
plt.plot(data[12,(0,4,1,5)],'-xb')
#plt.plot(data[12,(2,6,3,7)],'-ob')
plt.plot(data[13,(0,4,1,5)],'-xg')
#plt.plot(data[13,(2,6,3,7)],'-og')
plt.plot(data[14,(0,4,1,5)],'-xm')
#plt.plot(data[14,(0,4,1,5)],'-om')
plt.plot((0,1,2),data[15,(0,4,1)],'-xc')
#plt.plot((0,1,2),data[15,(2,6,3)],'-oc')
#plt.plot((0,2),data[16,(0,1)],'-xk')
plt.plot((0,1,2),data[16,(2,6,3)],'-ok')
plt.legend(hkl_yz_s,fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_YZ_group_last_first.png')
plt.show()


# # Relaxation strain -ZX


plt.figure(figsize=(15,10))
plt.plot(data[18,(0,4,1,5)],'-xr')
#plt.plot(data[18,(2,6,3,7)],'-or')
plt.plot(data[19,(0,4,1,5)],'-xb')
#plt.plot(data[19,(2,6,3,7)],'-ob')
plt.plot(data[20,(0,4,1,5)],'-xg')
#plt.plot(data[20,(2,6,3,7)],'-og')
plt.plot(data[21,(0,4,1,5)],'-xm')
#plt.plot(data[21,(0,4,1,5)],'-om')
plt.plot(data[22,(0,4,1,5)],'-xc')
#plt.plot(data[22,(2,6,3,7)],'-om')
plt.plot(data[23,(0,4,1,5)],'-xk')
#plt.plot(data[23,(2,6,3,7)],'-ok')
plt.legend(hkl_mzx_s,fancybox=True, loc='upper center',shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_mZX_group_last_first.png')
plt.show()




