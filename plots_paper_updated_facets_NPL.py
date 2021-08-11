#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:59:36 2021

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



filename = 'Delta_eps_facet_avg.txt' #'Delta_eps_ref_O2_3rd.txt'

#'Delta_eps_facet_sorted.txt'

pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_3/'

data = np.loadtxt(pathload+filename)
savedir = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_3/'


# Large crystal updated list



hkl_001 = ['[0 -1 0]','[0 0 1]','[1 0 0]','[-1 0 0]','[0 0 -1]','[0 1 0]','average']

hkl_111 = ['[1 -1 1]','[-1 -1 1]','[1 -1 -1]','[-1 1 1]','[1 1 -1]','[-1 1 -1]','average']
#[[1,-1,1],[-1,-1,1],[1,-1,-1],[1,1,1],[-1,-1,-1],[-1,1,1],[1,1,-1],[-1,1,-1]]

hkl_011 = ['[0,-1,1]','[1,-1,0]','[-1,0,1]','[1,0,-1]','[-1,1,0]','[0,1,-1]'
           ,'[1,0,1]','[0,1,1]','[1,1,0]','Average','{1 -1 0}','{1,1,0}']

hkl_1m10 = ['[0 -1 1]','[1 -1 0]','[-1 0 1]','[1 0 -1]','[-1 1 0]','[0 1 -1]']

hkl_110 =  ['[1 0 1]','[0 1 1]','[1 1 0]']




hkl_311 = ['[1,1,-3]','[-1,-1,3]','[-3,1,1]','[-1,3,-1]','[3,-1,-1]',
           '[-1,-3,1]','[1,-3,-1]','[1,-1,-3]','[-1,1,3]','[-3,-1,1]','[-1,1,-3]',
           '[1,-1,3]','[-1,3,1]','[1,3,-1]','[3,1,-1]','[3,-1,1]','[-3,1,-1]',
           '[1,1,3]','[1,3,1]','[3,1,1]','Average','{1 1 -3}','{1,-1,-3}','{1,1,3}']

hkl_11m3 = ['[1,1,-3]','[-1,-1,3]','[-3,1,1]','[-1,3,-1]','[3,-1,-1]']
hkl_1m1m3 = ['[-1 -3 1]','[1 -3 -1]','[1 -1 -3]','[-1 1 3]','[-3 -1 1]','[-1 1 -3]',
             '[1 -1 3]','[-1 3 1]','[1 3 -1]','[3 1 -1]','[3 -1 1]','[-3 1 -1]']
hkl_113 = ['[1 1 3]','[1 3 1]','[3 1 1]']






hkl_average = ['{1 0 0}','{1 1 1}','{1 -1 0}','{1 1 0}','{1 1 -3}','{1 -1 -3}','{1 1 3}']

# hkl_comb = ['{1,0,0}','{1,0,0}','{1,1,1}','{1,1,1}','{1,1,-3}','{1,1,-3}',
#             '{1,-1,-3}','{1,-1,-3}','{1,-1,0}','{1,-1,0}','{1,1,0}','{1,1,0}']


# Average values

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(data[6,0:4],'-o',linewidth=4)
plt.plot(data[13,0:4],'-o',linewidth=4)
plt.plot(data[24,0:4],'-o',linewidth=4)
plt.plot(data[25,0:4],'-o',linewidth=4)
plt.plot(data[47,0:4],'-o',linewidth=4)
plt.plot(data[48,0:4],'-o',linewidth=4)
plt.plot(data[49,0:4],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_average,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()
plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_average.png',bbox_inches='tight')



plt.show()


# Average delta strain 001 facets

plt.figure(figsize=(15,10))

ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)
    
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_001,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()
plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_001.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

start = 7

for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)
    
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_111,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_111.png')
plt.show()


# Average delta strain 1 -1 0 facets

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)
    
start = 14


for i in range(start,start+len(hkl_1m10)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_1m10,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_1m10.png')
plt.show()

# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)
start = 20


for i in range(start,start+len(hkl_110)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_110,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_110.png')
plt.show()


# Average delta strain 1 1 -3 facets


plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

start = 26

for i in range(start,start+len(hkl_11m3)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_11m3,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()

plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_11m3.png')
plt.show()


# Average delta strain 1 -1 -3 facets

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

start = 31

for i in range(start,start+len(hkl_1m1m3)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_1m1m3,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()


plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_1m1m3.png',bbox_inches='tight')
plt.show()


# Average delta strain 1 1 3 facets

plt.figure(figsize=(15,10))
ax = plt.subplot()

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)


start = 43

for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,0:4],'-o',linewidth=4)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_113,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':27})
plt.tight_layout()
plt.savefig(savedir+'Average_delta_strain_facet_crystal_3_113.png',bbox_inches='tight')
plt.show()