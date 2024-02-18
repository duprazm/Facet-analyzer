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


last_first = data[:,[8,12,15,20]]
average = data[:,[9,14,17,22]]


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

hkl_myz_s=['[0 1 0]','[-1 1 -1]','[-1 1 0]',
         '[0 1 -1]']

hkl_yz=[[0,-1,0],[0,-1,0],[1,-1,1],[1,-1,1],[-1,-1,1],[-1,-1,1],
        [0,-1,1],[0,-1,1]]#,[-1,-1,3],[-1,-1,3],[1,-3,1],[1,-3,1]]

hkl_yz_s=['[0 -1 0]','[1 -1 1]','[-1 -1 1]',
        '[0,-1,1]','[-1 -1 3]','[1 -3 1]']

hkl_mzx=[[1,0,0],[1,0,0],[0,0,-1],[0,0,-1],[1,-1,-1],[1,-1,-1],
        [1,0,-1],[1,0,-1],[1,-1,3],[1,-1,3],[1,1,-3],[1,1,-3]]

hkl_mzx_s=['[1 0 0]','[0 0 -1]','[1 -1 -1]',
        '[1 0 -1]','[1 -1 3]','[1 1 -3]']

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


############# Non stoichiometric conditions (cycle 1 and cycle 2) ############

# Average strain per facet group


plt.figure(figsize=(15,10))
ax = plt.subplot()


for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)


plt.plot(data[5,(0,4,1,5)],'-xr',linewidth=4)
#plt.plot(data[5,(2,6,3,7)],'-or')
plt.plot(data[10,(0,4,1,5)],'-xb',linewidth=4)
#plt.plot(data[10,(2,6,3,7)],'-ob')
plt.plot(data[17,(0,4,1,5)],'-xg',linewidth=4)
#plt.plot(data[17,(2,6,3,7)],'-og')
plt.plot(data[24,(0,4,1,5)],'-xm',linewidth=4)
#plt.plot(data[24,(2,6,3,7)],'-om')


for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_avg_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()


plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_group_last_first.png')
plt.show()

#Relaxation strain +ZX

plt.figure(figsize=(15,10))
ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(data[0,(0,4,1,5)],'-xr',linewidth=4)
#plt.plot(data[0,(2,6,3,7)],'-or')
plt.plot(data[1,(0,4,1,5)],'-xb',linewidth=4)
#plt.plot(data[1,(2,6,3,7)],'-ob')
#plt.plot((0,2,3),data[2,(0,1,5)],'-xg')
plt.plot(data[2,(2,6,3,7)],'-og',linewidth=4)
#plt.plot(data[3,(0,4,1,5)],'-xm')
plt.plot(2,data[3,3],'-om',linewidth=4)
#plt.plot(data[4,(0,4,1,5)],'-xc')
plt.plot(data[4,(2,6,3,7)],'-oc',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_zx_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_ZX_last_first.png')
plt.show()


#Relaxation strain -YZ


plt.figure(figsize=(15,10))
ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)


plt.plot(data[6,(0,4,1,5)],'-xr',linewidth=4)
#plt.plot(data[6,(2,6,3,7)],'-or')
plt.plot(data[7,(0,4,1,5)],'-xb',linewidth=4)
#plt.plot(data[7,(2,6,3,7)],'-ob')
plt.plot(data[8,(0,4,1,5)],'-xg',linewidth=4)
#plt.plot(data[8,(2,6,3,7)],'-og')
plt.plot(data[9,(0,4,1,5)],'-xm',linewidth=4)
#plt.plot(data[9,(2,6,3,7)],'-om')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_myz_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_mYZ_last_first.png')
plt.show()


# # Relaxation strain +YZ
plt.figure(figsize=(15,10))
ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)
    
    
#plt.plot((0,2,3),data[11,(0,1,5)],'-xr')
plt.plot(data[11,(2,6,3,7)],'-or',linewidth=4)
plt.plot(data[12,(0,4,1,5)],'-xb',linewidth=4)
#plt.plot(data[12,(2,6,3,7)],'-ob')
plt.plot(data[13,(0,4,1,5)],'-xg',linewidth=4)
#plt.plot(data[13,(2,6,3,7)],'-og')
plt.plot(data[14,(0,4,1,5)],'-xm',linewidth=4)
#plt.plot(data[14,(0,4,1,5)],'-om')
plt.plot((0,1,2),data[15,(0,4,1)],'-xc',linewidth=4)
#plt.plot((0,1,2),data[15,(2,6,3)],'-oc',linewidth=4)
#plt.plot((0,2),data[16,(0,1)],'-xk')
plt.plot((0,1,2),data[16,(2,6,3)],'-ok',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_yz_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()


plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_YZ_group_last_first.png')
plt.show()

# # Relaxation strain -ZX


plt.figure(figsize=(15,10))
ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)


plt.plot(data[18,(0,4,1,5)],'-xr',linewidth=4)
#plt.plot(data[18,(2,6,3,7)],'-or')
plt.plot(data[19,(0,4,1,5)],'-xb',linewidth=4)
#plt.plot(data[19,(2,6,3,7)],'-ob')
plt.plot(data[20,(0,4,1,5)],'-xg',linewidth=4)
#plt.plot(data[20,(2,6,3,7)],'-og')
plt.plot(data[21,(0,4,1,5)],'-xm',linewidth=4)
#plt.plot(data[21,(0,4,1,5)],'-om')
plt.plot(data[22,(0,4,1,5)],'-xc',linewidth=4)
#plt.plot(data[22,(2,6,3,7)],'-om')
plt.plot(data[23,(0,4,1,5)],'-xk',linewidth=4)
#plt.plot(data[23,(2,6,3,7)],'-ok')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_mzx_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()


plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_mZX_group_last_first.png')
plt.show()


##################### Stoichiometric conditions ###############################

# Average strain per facet group


plt.figure(figsize=(15,10))

ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)
    
plt.plot(last_first[5,:],'-xr',linewidth=4)
#plt.plot(average[5,:],'-or')
plt.plot(last_first[10,:],'-xb',linewidth=4)
#plt.plot(average[10,:],'-ob')
plt.plot(last_first[17,:],'-xg',linewidth=4)
#plt.plot(average[17,:],'-og')
plt.plot(last_first[24,:],'-xm',linewidth=4)
#plt.plot(average[24,:],'-om')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_avg_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_ref_02_4_group_last_first.png')
plt.show()


# Relaxation strain +ZX


plt.figure(figsize=(15,10))

ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)


plt.plot(last_first[0,:],'-xr',linewidth=4)
#plt.plot(average[0,:],'-or')
plt.plot(last_first[1,:],'-xb',linewidth=4)
#plt.plot(average[1,:],'-ob')
#plt.plot(last_first[2,0],'-xg')
plt.plot(average[2,:],'-og',linewidth=4)
#plt.plot(last_first[3,0:3],'-xm')
plt.plot(average[3,:],'-om',linewidth=4)
plt.plot(last_first[4,:],'-xc',linewidth=4)
#plt.plot(average[4,:],'-oc')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_zx_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_ref_02_4_ZX_group_last_first.png')
plt.show()


# Relaxation strain -YZ

plt.figure(figsize=(15,10))

ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(last_first[6,:],'-xr',linewidth=4)
#plt.plot(average[6,:],'-or')
plt.plot(last_first[7,:],'-xb',linewidth=4)
#plt.plot(average[7,:],'-ob')
plt.plot(last_first[8,:],'-xg',linewidth=4)
#plt.plot(average[8,:],'-og')
plt.plot(last_first[9,:],'-xm',linewidth=4)
#plt.plot(average[9,:],'-om')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_myz_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_ref_02_4_mYZ_group_last_first.png')
plt.show()

# Relaxation strain YZ

plt.figure(figsize=(15,10))

ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(last_first[11,:],'-xr',linewidth=4)
#plt.plot(average[11,:],'-or')
plt.plot(last_first[12,:],'-xb',linewidth=4)
#plt.plot(average[12,:],'-ob')
plt.plot(last_first[13,:],'-xg',linewidth=4)
#plt.plot(average[13,:],'-og')
plt.plot(last_first[14,:],'-xm',linewidth=4)
#plt.plot(average[14,:],'-om')
plt.plot(range(1,4),last_first[15,1:4],'-xc',linewidth=4)
#plt.plot(range(1,4),average[15,1:4],'-oc')
plt.plot(range(1,4),last_first[16,1:4],'-xk',linewidth=4)
#plt.plot(range(1,4),average[16,1:4],'-ok')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_yz_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_ref_02_4_YZ_group_last_first.png')
plt.show()


# # Relaxation strain -ZX

plt.figure(figsize=(15,10))

ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(last_first[18,:],'-xr',linewidth=4)
#plt.plot(average[18,:],'-or')
plt.plot(last_first[19,:],'-xb',linewidth=4)
#plt.plot(average[19,:],'-ob')
plt.plot(last_first[20,:],'-xg',linewidth=4)
#plt.plot(average[20,:],'-og')
plt.plot(last_first[21,:],'-xm',linewidth=4)
#plt.plot(average[21,:],'-om')
plt.plot(last_first[22,:],'-xc',linewidth=4)
#plt.plot(average[22,:],'-oc')
plt.plot(last_first[23,:],'-xk',linewidth=4)
#plt.plot(average[23,:],'-ok')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_mzx_s,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()


plt.savefig(savedir+'Average_delta_strain_ref_02_4_mZX_group.png')
plt.show()

