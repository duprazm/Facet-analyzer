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

#'Delta_eps.txt'

pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

data = np.loadtxt(pathload+filename)
savedir = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

# Relaxation displacement last disp vs average, average facets

# Ref 3rd O2

# last_first = data[:,[0,3,6]]
# last_last = data[:,[1,4,7]]
# average = data[:,[2,5,8]]

# Ref 4th O2

last_first = data[:,[8,12,15,20]]
average = data[:,[9,14,17,22]]

last_first_O = data[:,[35,12,15,20]]
average_O = data[:,[37,14,17,22]]

hkl_avg = ['ZX','ZX','-YZ','-YZ','YZ','YZ','-ZX','-ZX']

hkl_avg_s = ['ZX','-YZ','YZ','-ZX']

hkl_zx =[[-1,0,0],[-1,0,0],[-1,1,1],[-1,1,1],[-1,1,3],[-1,1,3],
         [-3,1,1],[-3,1,1],[-1,0,1],[-1,0,1]]

hkl_zx_s = [[-1,0,0],[-1,1,1],[-1,1,3],
         [-3,1,1],[-1,0,1]] 

hkl_myz=[[0,1,0],[0,1,0],[-1,1,-1],[-1,1,-1],[-1,1,0],[-1,1,0],
         [0,1,-1],[0,1,-1]]

hkl_myz_s = [[0,1,0],[-1,1,-1],[-1,1,0],
         [0,1,-1]] 

hkl_yz=[[0,-1,0],[0,-1,0],[1,-1,1],[1,-1,1],[-1,-1,1],[-1,-1,1],
        [0,-1,1],[0,-1,1]]#,[-1,-1,3],[-1,-1,3],[1,-3,1],[1,-3,1]]

hkl_yz_s=[[0,-1,0],[1,-1,1],[-1,-1,1],[0,-1,1],[-1,-1,3],[1,-3,1]]

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

# ref O2 4th cycle + ref O2 3rd cycle CO:O2

plt.figure(figsize=(15,10))

plt.plot(last_first[5,:],'-xr')
#plt.plot(average[5,:],'-or')
plt.plot(last_first[10,:],'-xb')
#plt.plot(average[10,:],'-ob')
plt.plot(last_first[17,:],'-xg')
#plt.plot(average[17,:],'-og')
plt.plot(last_first[24,:],'-xm')
#plt.plot(average[24,:],'-om')
plt.legend(hkl_avg_s, fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group_last_first.png')
plt.show()


# ref O2 4th cycle 

plt.figure(figsize=(15,10))

plt.plot(last_first_O[5,:],'-xr')
plt.plot(average_O[5,:],'-or')
plt.plot(last_first_O[10,:],'-xb')
plt.plot(average_O[10,:],'-ob')
plt.plot(last_first_O[17,:],'-xg')
plt.plot(average_O[17,:],'-og')
plt.plot(last_first_O[24,:],'-xm')
plt.plot(average_O[24,:],'-om')
plt.legend(hkl_avg, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'average_O_delta_strain_ref_02_4_init_group.png')
plt.show()

# ref O2 3rd cycle

plt.figure(figsize=(15,10))

plt.plot(last_first[5,:],'-xr')
plt.plot(average[5,:],'-or')
plt.plot(last_first[10,:],'-xb')
plt.plot(average[10,:],'-ob')
plt.plot(last_first[17,:],'-xg')
plt.plot(average[17,:],'-og')
plt.plot(last_first[24,:],'-xm')
plt.plot(average[24,:],'-om')
plt.legend(hkl_avg, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_group.png')
plt.show()

# ref O2 3rd cycle + comparison with O2 4th cycle average


plt.figure(figsize=(15,10))

plt.plot(data[5,(0,3,6,11)],'-xr')
plt.plot(data[5,(2,5,8,13)],'-or')
plt.plot(data[10,(0,3,6,11)],'-xb')
plt.plot(data[10,(2,5,8,13)],'-ob')
plt.plot(data[17,(0,3,6,11)],'-xg')
plt.plot(data[17,(2,5,8,13)],'-og')
plt.plot(data[24,(0,3,6,11)],'-xm')
plt.plot(data[24,(2,5,8,13)],'-om')
plt.legend(hkl_avg, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_3_4_group.png')
plt.show()



#Relaxation strain +ZX

plt.figure(figsize=(15,10))
plt.plot(last_first[0,:],'-xr')
#plt.plot(average[0,:],'-or')
plt.plot(last_first[1,:],'-xb')
#plt.plot(average[1,:],'-ob')
#plt.plot(last_first[2,0],'-xg')
plt.plot(average[2,:],'-og')
#plt.plot(last_first[3,0:3],'-xm')
plt.plot(average[3,:],'-om')
plt.plot(last_first[4,:],'-xc')
#plt.plot(average[4,:],'-oc')
plt.legend(hkl_zx_s,fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_ZX_group_last_first.png')
plt.show()

# ref O2 4th cycle 

plt.figure(figsize=(15,10))
plt.plot(last_first_O[0,:],'-xr')
plt.plot(average_O[0,:],'-or')
plt.plot(last_first_O[1,:],'-xb')
plt.plot(average_O[1,:],'-ob')
plt.plot(last_first_O[2,0],'-xg')
plt.plot(average_O[2,:],'-og')
plt.plot(last_first_O[3,0:3],'-xm')
plt.plot(average_O[3,:],'-om')
plt.plot(last_first_O[4,:],'-xc')
plt.plot(average_O[4,:],'-oc')
plt.legend(hkl_zx,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'average_O_delta_strain_ref_02_4_init_ZX_group.png')
plt.show()

# ref O2 3rd cycle

plt.figure(figsize=(15,10))
plt.plot(last_first[0,:],'-xr')
plt.plot(average[0,:],'-or')
plt.plot(last_first[1,:],'-xb')
plt.plot(average[1,:],'-ob')
plt.plot(last_first[2,0],'-xg')
plt.plot(average[2,:],'-og')
plt.plot(last_first[3,0:3],'-xm')
plt.plot(average[3,:],'-om')
plt.plot(last_first[4,:],'-xc')
plt.plot(average[4,:],'-oc')
plt.legend(hkl_zx,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_ZX_group.png')
plt.show()

# ref O2 3rd cycle + comparison with O2 4th cycle average

plt.figure(figsize=(15,10))
plt.plot(data[0,(2,5,8,13)],'-or')
plt.plot(data[1,(2,5,8,13)],'-ob')
plt.plot(data[2,(2,5,8,13)],'-og')
plt.plot(data[3,(2,5,8,13)],'-om')
plt.plot(data[4,(2,5,8,13)],'-oc')
plt.legend(hkl_zx_s,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_4_ZX_group_average.png')
plt.show()

# # Relaxation strain -YZ

plt.figure(figsize=(15,10))
plt.plot(last_first[6,:],'-xr')
#plt.plot(average[6,:],'-or')
plt.plot(last_first[7,:],'-xb')
#plt.plot(average[7,:],'-ob')
plt.plot(last_first[8,:],'-xg')
#plt.plot(average[8,:],'-og')
plt.plot(last_first[9,:],'-xm')
#plt.plot(average[9,:],'-om')
plt.legend(hkl_myz,fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_mYZ_group_last_first.png')
plt.show()

# ref O2 4th cycle

plt.figure(figsize=(15,10))
plt.plot(last_first_O[6,:],'-xr')
plt.plot(average_O[6,:],'-or')
plt.plot(last_first_O[7,:],'-xb')
plt.plot(average_O[7,:],'-ob')
plt.plot(last_first_O[8,:],'-xg')
plt.plot(average_O[8,:],'-og')
plt.plot(last_first_O[9,:],'-xm')
plt.plot(average_O[9,:],'-om')
plt.legend(hkl_myz,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_init_mYZ_group.png')
plt.show()


# ref O2 3rd cycle

plt.figure(figsize=(15,10))
plt.plot(last_first[6,:],'-xr')
plt.plot(average[6,:],'-or')
plt.plot(last_first[7,:],'-xb')
plt.plot(average[7,:],'-ob')
plt.plot(last_first[8,:],'-xg')
plt.plot(average[8,:],'-og')
plt.plot(last_first[9,:],'-xm')
plt.plot(average[9,:],'-om')
plt.legend(hkl_myz,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_mYZ_group.png')
plt.show()

# ref O2 3rd cycle + comparison with O2 4th cycle average


plt.figure(figsize=(15,10))
plt.plot(data[6,(2,5,8,13)],'-or')
plt.plot(data[7,(2,5,8,13)],'-ob')
plt.plot(data[8,(2,5,8,13)],'-og')
plt.plot(data[9,(2,5,8,13)],'-om')
plt.legend(hkl_myz_s,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_4_mYZ_group.png')
plt.show()


# # Relaxation strain +YZ

plt.figure(figsize=(15,10))
plt.plot(last_first[11,:],'-xr')
#plt.plot(average[11,:],'-or')
plt.plot(last_first[12,:],'-xb')
#plt.plot(average[12,:],'-ob')
plt.plot(last_first[13,:],'-xg')
#plt.plot(average[13,:],'-og')
plt.plot(last_first[14,:],'-xm')
#plt.plot(average[14,:],'-om')
plt.plot(range(1,4),last_first[15,1:4],'-xc')
#plt.plot(range(1,4),average[15,1:4],'-oc')
plt.plot(range(1,4),last_first[16,1:4],'-xk')
#plt.plot(range(1,4),average[16,1:4],'-ok')
plt.legend(hkl_yz_s,fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_YZ_group_last_first.png')
plt.show()



# ref O2 4th cycle

plt.figure(figsize=(15,10))
plt.plot(last_first_O[11,:],'-xr')
plt.plot(average_O[11,:],'-or')
plt.plot(last_first_O[12,:],'-xb')
plt.plot(average_O[12,:],'-ob')
plt.plot(last_first_O[13,:],'-xg')
plt.plot(average_O[13,:],'-og')
plt.plot(last_first_O[14,:],'-xm')
plt.plot(average_O[14,:],'-om')
plt.plot(range(1,4),last_first_O[15,1:4],'-xc')
plt.plot(range(1,4),average_O[15,1:4],'-oc')
plt.plot(range(1,4),last_first_O[16,1:4],'-xk')
plt.plot(range(1,4),average_O[16,1:4],'-ok')
plt.legend(hkl_yz,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'average_O_delta_strain_ref_02_4_i_init_YZ_group.png')
plt.show()

# ref O2 3rd cycle

plt.figure(figsize=(15,10))
plt.plot(last_first[11,:],'-xr')
plt.plot(average[11,:],'-or')
plt.plot(last_first[12,:],'-xb')
plt.plot(average[12,:],'-ob')
plt.plot(last_first[13,:],'-xg')
plt.plot(average[13,:],'-og')
plt.plot(last_first[14,:],'-xm')
plt.plot(average[14,:],'-om')
plt.legend(hkl_yz,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_YZ_group.png')
plt.show()

# ref O2 3rd cycle + comparison with O2 4th cycle average

plt.figure(figsize=(15,10))
plt.plot(data[11,(2,5,8,13)],'-or')
plt.plot(data[12,(2,5,8,13)],'-ob')
plt.plot(data[13,(2,5,8,13)],'-og')
plt.plot(data[14,(2,5,8,13)],'-om')
plt.legend(hkl_yz_s,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_4_YZ_group.png')
plt.show()


# # Relaxation strain -ZX

plt.figure(figsize=(15,10))
plt.plot(last_first[18,:],'-xr')
#plt.plot(average[18,:],'-or')
plt.plot(last_first[19,:],'-xb')
#plt.plot(average[19,:],'-ob')
plt.plot(last_first[20,:],'-xg')
#plt.plot(average[20,:],'-og')
plt.plot(last_first[21,:],'-xm')
#plt.plot(average[21,:],'-om')
plt.plot(last_first[22,:],'-xc')
#plt.plot(average[22,:],'-oc')
plt.plot(last_first[23,:],'-xk')
#plt.plot(average[23,:],'-ok')
plt.legend(hkl_mzx_s,fancybox=True, shadow=True,prop={'size': 20})
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_mZX_group.png')
plt.show()


# ref O2 4th cycle

plt.figure(figsize=(15,10))
plt.plot(last_first_O[18,:],'-xr')
plt.plot(average_O[18,:],'-or')
plt.plot(last_first_O[19,:],'-xb')
plt.plot(average_O[19,:],'-ob')
plt.plot(last_first_O[20,:],'-xg')
plt.plot(average_O[20,:],'-og')
plt.plot(last_first_O[21,:],'-xm')
plt.plot(average_O[21,:],'-om')
plt.plot(last_first_O[22,:],'-xc')
plt.plot(average_O[22,:],'-oc')
plt.plot(last_first_O[23,:],'-xk')
plt.plot(average_O[23,:],'-ok')
plt.legend(hkl_mzx,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'average_O_delta_strain_ref_02_4_init_mZX_group.png')
plt.show()

# ref O2 3rd cycle

plt.figure(figsize=(15,10))
plt.plot(last_first[18,:],'-xr')
plt.plot(average[18,:],'-or')
plt.plot(last_first[19,:],'-xb')
plt.plot(average[19,:],'-ob')
plt.plot(last_first[20,:],'-xg')
plt.plot(average[20,:],'-og')
plt.plot(last_first[21,:],'-xm')
plt.plot(average[21,:],'-om')
plt.plot(last_first[22,:],'-xc')
plt.plot(average[22,:],'-oc')
plt.plot(last_first[23,:],'-xk')
plt.plot(average[23,:],'-ok')
plt.legend(hkl_mzx,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_mZX_group.png')
plt.show()

# ref O2 3rd cycle + comparison with O2 4th cycle average

plt.figure(figsize=(15,10))
plt.plot(data[18,(2,5,8,13)],'-or')
plt.plot(data[19,(2,5,8,13)],'-ob')
plt.plot(data[20,(2,5,8,13)],'-og')
plt.plot(data[21,(2,5,8,13)],'-om')
plt.plot(data[22,(2,5,8,13)],'-oc')
plt.plot(data[23,(2,5,8,13)],'-ok')
plt.legend(hkl_mzx_s,fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_3_4_mZX_group.png')
plt.show()

