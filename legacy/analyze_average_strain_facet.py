#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:44:31 2021

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

filename = 'Delta_eps_facet_sorted.txt' #'Delta_eps_ref_O2_3rd.txt'

#'Delta_eps_facet_sorted.txt'

pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'

data = np.loadtxt(pathload+filename)
savedir = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/tol_0_164/'


# Small crystal updated list



hkl_001 = ['[0,-1,0]','[0,0,1]','[1,0,0]','[-1,0,0]','[0,0,-1]','[0,1,0]','average']

hkl_111 = ['[1,-1,1]','[-1,-1,1]','[1,-1,-1]','[-1,1,1]','[1,1,-1]','[-1,1,-1]','average']
#[[1,-1,1],[-1,-1,1],[1,-1,-1],[1,1,1],[-1,-1,-1],[-1,1,1],[1,1,-1],[-1,1,-1]]

hkl_113 = ['[1,-3,1]','[1,1,-3]','[-1,-1,3]','[-3,1,1]','[1,-1,-3]','[-1,1,3]',
           '[-1,-3,1]','[1,-3,-1]','[-3,-1,1]','Average','{1 1 -3}','{1 -1 -3}']

hkl_11m3 = ['[1,-3,1]','[1,1,-3]','[-1,-1,3]','[-3,1,1]']
hkl_1m1m3 = ['[1,-1,-3]','[-1,1,3]','[-1,-3,1]','[1,-3,-1]'] #,'[-3,-1,1]']

hkl_011 = ['[0,-1,1]','[1,-1,0]','[-1,0,1]','[1,0,-1]','[-1,1,0]','[0,1,-1]'
           ,'[1,0,1]','[0,1,1]','[1,1,0]','Average','{1 -1 0}','{1,1,0}']

hkl_1m10 = ['[0,-1,1]','[1,-1,0]','[-1,0,1]','[1,0,-1]','[-1,1,0]','[0,1,-1]']

hkl_110 =  ['[1,0,1]','[0,1,1]','[1,1,0]']


hkl_average = ['{1,0,0}','{1,1,1}','{1,1,3}','{1,-1,-3}','{1,-1,0}','{1,1,0}']

hkl_average_p = ['{1,0,0}','{1,1,1}','{1,-1,0}','{1,1,0}','{1,1,-3}','{1,-1,-3}']

hkl_comb = ['{1,0,0}','{1,0,0}','{1,1,1}','{1,1,1}','{1,1,-3}','{1,1,-3}',
            '{1,-1,-3}','{1,-1,-3}','{1,-1,0}','{1,-1,0}','{1,1,0}','{1,1,0}']

############################### 1st cycle #####################################

# Average delta strain 001 facets

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_001.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
legend = hkl_111
start = 7


for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5)],'-o')
    plt.legend(hkl_111, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()


# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_011
start = 26


for i in range(start,start+len(hkl_011)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5)],'-o')
    plt.legend(hkl_011, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()

# Average delta strain 1 1 3 facets

plt.figure(figsize=(15,10))
legend = hkl_113
start = 14

for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5)],'-o')
    plt.legend(hkl_113, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()


# Average values 

plt.figure(figsize=(15,10))
legend = hkl_average
plt.plot(data[6,(0,5)],'-o')
plt.plot(data[13,(0,5)],'-o')
plt.plot(data[24,(0,5)],'-o')
plt.plot(data[25,(0,5)],'-o')
plt.plot(data[36,(0,5)],'-o')
plt.plot(data[37,(0,5)],'-o')
plt.legend(hkl_average, fancybox=True, shadow=True)
plt.show()


############################### 2nd cycle #####################################

# Average delta strain 001 facets

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(1,6)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
legend = hkl_111
start = 7


for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(1,6)],'-o')
    plt.legend(hkl_111, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()


# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_011
start = 26


for i in range(start,start+len(hkl_011)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(1,6)],'-o')
    plt.legend(hkl_011, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()

# Average delta strain 1 1 3 facets

plt.figure(figsize=(15,10))
legend = hkl_113
start = 14

for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(1,6)],'-o')
    plt.legend(hkl_113, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_group.png')
plt.show()


############################### 1st and 2nd cycle #############################

# Average delta strain 001 facets

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_001.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
legend = hkl_111
start = 7


for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_111, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_111.png')
plt.show()


# Average delta strain all 1 1 0  facets

plt.figure(figsize=(15,10))
legend = hkl_011
start = 26


for i in range(start,start+len(hkl_011)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_011, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_011_tot.png')
plt.show()

# Separate 1 -1 0 from 1 1 0


# Average delta strain 1 -1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_1m10
start = 26


for i in range(start,start+len(hkl_1m10)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_1m10, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_1m10.png')
plt.show()

# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_110
start = 32

for i in range(start,start+len(hkl_110)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_110, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_110.png')
plt.show()


# Average delta strain all 1 1 3 facets

plt.figure(figsize=(15,10))
legend = hkl_113
start = 14
for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_113, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_113_tot.png')
plt.show()


# Average delta strain 1 1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_11m3
start = 14

# for i in range(start,start+len(hkl_11m3)):
#     #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
#     plt.plot(data[i,(0,5,1,6)],'-o')
#     plt.legend(hkl_11m3, fancybox=True, shadow=True)
#     plt.xlabel('scan')
#     plt.ylabel('eps_111(x10⁴)')
plt.plot(range(3),data[14,(0,5,1)],'-o')
plt.plot(range(4),data[15,(0,5,1,6)],'-o')
plt.plot(range(3),data[16,(0,5,1)],'-o')
plt.plot(2,data[17,1],'-o')
plt.legend(hkl_113, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_11m3.png')
plt.show()

# Average delta strain 1 -1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_1m1m3
start = 18

for i in range(start,start+len(hkl_1m1m3)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(0,5,1,6)],'-o')
    plt.legend(hkl_1m1m3, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_1m1m3.png')
plt.show()

# Average delta strain combined

# Figure paper



fig=plt.figure(figsize=(15,10))
ax = plt.subplot()
#legend = hkl_average

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(3)

plt.plot(data[6,(0,5,1,6)],'-o',linewidth=4)
plt.plot(data[13,(0,5,1,6)],'-o',linewidth=4)
plt.plot(data[36,(0,5,1,6)],'-o',linewidth=4)
plt.plot(data[37,(0,5,1,6)],'-o',linewidth=4)
plt.plot(data[24,(0,5,1,6)],'-o',linewidth=4)
plt.plot(data[25,(0,5,1,6)],'-o',linewidth=4)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)

ax.legend(hkl_average_p,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, shadow=True, prop={'size':28})
#plt.savefig(savedir+'Average_delta_strain_cycle_1_cycle_2_all.png')
plt.show()



############################# 3rd and 4th cycle #######################################


# Average delta strain 001 facets

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_001.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
legend = hkl_111
start = 7


for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_111, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_111.png')
plt.show()


# Average delta strain all 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_011
start = 26


for i in range(start,start+len(hkl_011)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_011, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_group_110_tot.png')
plt.show()


# Average delta strain 1 -1 0 facets


plt.figure(figsize=(15,10))
legend = hkl_1m10
start = 26


for i in range(start,start+len(hkl_1m10)):    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_1m10, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_1m10.png')
plt.show()

# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_110
start = 32

for i in range(start,start+len(hkl_110)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_110, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_110.png')
plt.show()


# Average delta strain all 1 1 3 facets

plt.figure(figsize=(15,10))
legend = hkl_113
start = 14

for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,7,3)],'-o')
    plt.legend(hkl_113, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_113_tot.png')
plt.show()


# Average delta strain 1 1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_11m3
start = 14

# for i in range(start,start+len(hkl_11m3)):
#     #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
#     plt.plot(data[i,(0,5,1,6)],'-o')
#     plt.legend(hkl_11m3, fancybox=True, shadow=True)
#     plt.xlabel('scan')
#     plt.ylabel('eps_111(x10⁴)')
plt.plot(2,data[14,3],'-o')
plt.plot(range(3),data[15,(2,7,3)],'-o')
plt.plot(2,data[16,3],'-o')
plt.plot((1,2),data[17,(7,3)],'-o')
plt.legend(hkl_113, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_11m3.png')
plt.show()


# Average delta strain 1 -1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_1m1m3
start = 18

plt.plot(range(3),data[18,(2,7,3)],'-o')
plt.plot(range(3),data[19,(2,7,3)],'-o')
plt.plot(2,data[20,3],'-o')
plt.plot(range(3),data[21,(2,7,3)],'-o')
plt.legend(hkl_1m1m3, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_cycle_3_cycle_4_1m1m3.png')
plt.show()



################# Strain relaxation after CO oxidation stoichio ###############


## Ref Oxygen 3rd cycle

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(2,5,8)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_3_001.png')
plt.show()



## Ref Oxygen 4th cycle -> strain relaxation

# Average delta strain 001 facets

plt.figure(figsize=(15,10))
legend = hkl_001

for i in range(len(hkl_001)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_001, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_001.png')
plt.show()


# Average delta strain 1 1 1 facets

plt.figure(figsize=(15,10))
legend = hkl_111
start = 7


for i in range(start,start+len(hkl_111)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_111, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_111.png')
plt.show()


# Average delta strain 1 1 0 facets

plt.figure(figsize=(15,10))
legend = hkl_011
start = 26


for i in range(start,start+len(hkl_011)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_011, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_110_tot.png')
plt.show()


# Average delta strain 1 -1 0 facets


plt.figure(figsize=(15,10))
legend = hkl_1m10
start = 26


for i in range(start,start+len(hkl_1m10)):    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_1m10, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_1m10.png')
plt.show()


# Average strain 1 1 0 facet

plt.figure(figsize=(15,10))
legend = hkl_110
start = 32

for i in range(start,start+len(hkl_110)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_110, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_110.png')
plt.show()


# Average delta strain 1 1 3 facets

plt.figure(figsize=(15,10))
legend = hkl_113
start = 14

for i in range(start,start+len(hkl_113)):
    #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
    plt.plot(data[i,(7,14,15,16)],'-o')
    plt.legend(hkl_113, fancybox=True, shadow=True)
    plt.xlabel('scan')
    plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_113_tot.png')
plt.show()


# Average delta strain 1 1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_11m3
start = 14

# for i in range(start,start+len(hkl_11m3)):
#     #legend = ' '.join(str('{:.0f}'.format(e)) for e in hkl_001[i])
    
#     plt.plot(data[i,(0,5,1,6)],'-o')
#     plt.legend(hkl_11m3, fancybox=True, shadow=True)
#     plt.xlabel('scan')
#     plt.ylabel('eps_111(x10⁴)')
plt.plot((1,2,3),data[14,(14,15,16)],'-o')
plt.plot(range(4),data[15,(7,14,15,16)],'-o')
plt.plot((1,2,3),data[16,(14,15,16)],'-o')
plt.plot(range(4),data[17,(7,14,15,16)],'-o')
plt.legend(hkl_113, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_11m3.png')
plt.show()

# Average delta strain 1 -1 -3 facets

plt.figure(figsize=(15,10))
legend = hkl_1m1m3
start = 18

plt.plot(range(4),data[18,(7,14,15,16)],'-o')
plt.plot(range(4),data[19,(7,14,15,16)],'-o')
plt.plot((1,2,3),data[20,(14,15,16)],'-o')
plt.plot(range(4),data[21,(7,14,15,16)],'-o')
plt.legend(hkl_113, fancybox=True, shadow=True)
plt.xlabel('scan')
plt.ylabel('eps_111(x10⁴)')
plt.savefig(savedir+'Average_delta_strain_ref_02_4_1m1m3.png')
plt.show()

# Average delta strain per facet family



plt.figure(figsize=(15,10))
#legend = hkl_average
plt.plot(data[6,(7,14,15,16)],'-or')
plt.plot(data[13,(7,14,15,16)],'-ob')
plt.plot(data[24,(7,14,15,16)],'-og')
plt.plot(data[25,(7,14,15,16)],'-oc')
plt.plot(data[36,(7,14,15,16)],'-om')
plt.plot(data[37,(7,14,15,16)],'-ok')
plt.legend(hkl_average, fancybox=True, shadow=True,prop={'size':20})
#plt.savefig(savedir+'Average_delta_strain_ref_02_4_all.png')
plt.grid()
plt.show()




############################ Average values ##################################

legend_comb = hkl_comb

plt.figure(figsize=(15,10))
#legend = hkl_average
plt.plot(data[6,(0,5,1,6)],'-or')
#plt.plot(range(2,4),data[6,(1,6)],'-xr')
plt.plot(range(4,7),data[6,(2,7,3)],'-+r')
plt.plot(data[13,(0,5,1,6)],'-ob')
#plt.plot(range(2,4),data[13,(1,6)],'-xb')
plt.plot(range(4,7),data[13,(2,7,3)],'-+b')
plt.plot(data[24,(0,5,1,6)],'-og')
#plt.plot(range(2,4),data[24,(1,6)],'-xg')
plt.plot(range(4,7),data[24,(2,7,3)],'-+g')
plt.plot(data[25,(0,5,1,6)],'-oc')
#plt.plot(range(2,4),data[25,(1,6)],'-xc')
plt.plot(range(4,7),data[25,(2,7,3)],'-+c')
plt.plot(data[36,(0,5,1,6)],'-om')
#plt.plot(range(2,4),data[36,(1,6)],'-xm')
plt.plot(range(4,7),data[36,(2,7,3)],'-+m')
plt.plot(data[37,(0,5,1,6)],'-ok')
#plt.plot(range(2,4),data[37,(1,6)],'-xk')
plt.plot(range(4,7),data[37,(2,7,3)],'-+k')
plt.legend(hkl_comb, fancybox=True, shadow=True)
#plt.savefig(savedir+'Average_delta_strain_cycle_1_4_all.png')
plt.show()



# Evolution during CO oxidation- ref Oxygen 4th cycle

plt.figure(figsize=(15,10))
legend = hkl_average
plt.plot(data[6,(7,14,15,16)],'-or')
plt.plot(data[13,(7,14,15,16)],'-ob')
plt.plot(data[24,(7,14,15,16)],'-og')
plt.plot(data[25,(7,14,15,16)],'-oc')
plt.plot(data[36,(7,14,15,16)],'-om')
plt.plot(data[37,(7,14,15,16)],'-ok')
plt.legend(hkl_average, fancybox=True, shadow=True)
plt.show()


# Evolution during CO oxidation - ref Oxygen 3rd cycle

plt.figure(figsize=(15,10))
legend = hkl_average
plt.plot(data[6,(7,14,15,16)],'-or')
plt.plot(data[13,(7,14,15,16)],'-ob')
plt.plot(data[24,(7,14,15,16)],'-og')
plt.plot(data[25,(7,14,15,16)],'-oc')
plt.plot(data[36,(7,14,15,16)],'-om')
plt.plot(data[37,(7,14,15,16)],'-ok')
plt.legend(hkl_average, fancybox=True, shadow=True)
plt.show()



