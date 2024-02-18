#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:34:38 2020

@author: maxime
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



pathload = '/users/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_4/updated_scans/tol_0_164/'
#'/home/maxime/Documents/maxime/Reseau/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'
#'/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'

# scan numbers -- crystal "small" P10

hkls =[[0,-1,1],[1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[0,1,1],[1,1,0],[-1,1,0],[0,1,-1]]


ref = [1,1,1]

angles=[]
## Calculation interplanar angles

for i,R in enumerate(hkls):
    angles.append(np.arccos(np.dot([hkls[i][0],hkls[i][1],hkls[i][2]]
                                      /norm([hkls[i][0],hkls[i][1],hkls[i][2]])
                                      ,ref/norm(ref)))*180/np.pi)
    print(str(hkls[i]) + ':' + str(angles[i]))


# Small crystal updated list

#[[1,-1,1],[-1,-1,1],[1,-1,-1],[-1,1,1],[1,1,-1],[-1,1,-1]]
#[[1,-1,1],[-1,-1,1],[1,-1,-1],[1,1,1],[-1,-1,-1],[-1,1,1],[1,1,-1],[-1,1,-1]]

# [[0,-1,0],[0,0,1],[1,0,0],[-1,0,0],[0,0,-1],[0,1,0]]

# [[1,-3,1],[-1,-3,1],[1,-3,-1],[-1,-1,3],[-3,-1,1],[-1,1,3],[1,-1,-3],[-3,1,1],[1,1,-3]]

# [[0,-1,1],[1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[0,1,1],[1,1,0],[-1,1,0],[0,1,-1]]



# Large crystal

#[[1, 1, 3],[1, -1, 3],[-1, -1, 3],[-1, 1, 3]] 
#[[1, 3, 1],[-1, 3, 1],[-1, 3, -1],[1, 3, -1]] 
#[[3, 1, 1],[3, 1, -1],[3, -1, -1],[3, -1, 1]] 
#[[-3, 1, -1],[-3, -1, 1],[-3, 1, 1],[-1, 3, 1]] 
#[[1, -3, -1],[1, -1, -3],[1, 1, -3],[-1, 1, -3]] 
#[[-1, -1, -1],[1, 1, 1]]
#[[-1, 1, -1],[-1, -1, 1],[1, -1, -1]] 
#[[1, -1, 1],[1, 1, -1],[-1, 1, 1]] 
#[[1, 0, 0],[0, 1, 0],[0, 0, 1]]  
#[[-1, 0, 0],[0, -1, 0],[0, 0, -1]] 
#[[-1 , 1, 0],[-1, 0, 1],[0, -1, 1]] 
#[[1 , -1, 0],[1, 0, -1],[0, 1, -1]] 
#[[0 , 1, 1],[1, 1, 0],[1, 0, 1]] 

datas=[]



# for i in range(len(hkls)):
#     filename = 'strain_disp_' + ' '.join(str('{:.0f}'.format(e)) for e in hkls[i]) + '.dat'

#     data = np.loadtxt(pathload+filename,skiprows=1,usecols=(0,1,2,3,4))
    
#     datas= np.append(datas,data)
#     datas = np.reshape(datas,(int(len(datas)/len(data[0])),len(data[0])))
    
#     print('Delta_U_O2_cycle1' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[3,1]-data[2,1])*10**0)))
#     print('Delta_U_O2_cycle2' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[9,1]-data[8,1])*10**0)))
#     print('Delta_U_O2_cycle3' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[12,1]-data[11,1])*10**0)))
#     print('Delta_U_O2_cycle4' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[20,1]-data[19,1])*10**0)))
    
# plt.figure(figsize=(15,10))
# for i in range(len(hkls)):    
    
#     legend = ' '.join(str('{:.0f}'.format(e)) for e in hkls[i])


#     plt.errorbar(datas[i*len(data[:]):(i+1)*len(data[:]),0],datas[i*len(data[:]):(i+1)*len(data[:]),1],datas[i*len(data[:]):(i+1)*len(data[:]),2], fmt='o',capsize=2,label=legend)
#     #plt.xlabel('Angle (deg.)')6
#     plt.ylabel('Retrieved <disp>')
#     plt.ylim(-1.4,1.4)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
#     plt.savefig(pathload + 'disp_' + str(hkls[0]) + '.png')
#     plt.grid()
# plt.show()

    
# plt.figure(figsize=(15,10))
# for i in range(len(hkls)): 
    
#     legend = ' '.join(str('{:.0f}'.format(e)) for e in hkls[i])

#     plt.errorbar(datas[i*len(data[:]):(i+1)*len(data[:]),0],datas[i*len(data[:]):(i+1)*len(data[:]),3],datas[i*len(data[:]):(i+1)*len(data[:]),4], fmt='o',capsize=2,label=legend)
#     #plt.xlabel('Angle (deg.)')6
#     plt.ylabel('Retrieved <strain>')
#     plt.ylim(-0.0006,0.0006)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
#     plt.savefig(pathload + 'strain_' + str(hkls[0]) + '.png')
#     #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
#     plt.grid()
# plt.show()

## Absolute strain values

for i in range(len(hkls)):
    filename = 'strain_disp_' + ' '.join(str('{:.0f}'.format(e)) for e in hkls[i]) + '.dat'

    data = np.loadtxt(pathload+filename,skiprows=1,usecols=(0,1,2,3,4))
    
    datas= np.append(datas,data)
    datas = np.reshape(datas,(int(len(datas)/len(data[0])),len(data[0])))
    
    print('U_O2_cycle1' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[15,1])*10**0)))
    print('U_O2_cycle2' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[16,1])*10**0)))
    #print('U_O2_cycle3' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[17,1])*10**0)))
    #print('U_O2_cycle4' + str(hkls[i]) + ':' , str('{:.2f}'.format((data[19,1])*10**0)))
    
plt.figure(figsize=(15,10))
for i in range(len(hkls)):    
    
    legend = ' '.join(str('{:.0f}'.format(e)) for e in hkls[i])


    plt.errorbar(datas[i*len(data[:]):(i+1)*len(data[:]),0],datas[i*len(data[:]):(i+1)*len(data[:]),1],datas[i*len(data[:]):(i+1)*len(data[:]),2], fmt='o',capsize=2,label=legend)
    #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <disp>')
    plt.ylim(-1.4,1.4)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    plt.savefig(pathload + 'disp_' + str(hkls[0]) + '.png')
    plt.grid()
plt.show()

    
plt.figure(figsize=(15,10))
for i in range(len(hkls)): 
    
    legend = ' '.join(str('{:.0f}'.format(e)) for e in hkls[i])

    plt.errorbar(datas[i*len(data[:]):(i+1)*len(data[:]),0],datas[i*len(data[:]):(i+1)*len(data[:]),3],datas[i*len(data[:]):(i+1)*len(data[:]),4], fmt='o',capsize=2,label=legend)
    #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.ylim(-0.0006,0.0006)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    plt.savefig(pathload + 'strain_' + str(hkls[0]) + '.png')
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.show()

