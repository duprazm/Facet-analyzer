# -*- coding: utf-8 -*-
########################################################################################################################
########################################################################################################################
# Extract strain at facets
# Need a vtk file extracted from the FacetAnalyser plugin of ParaView (information: point data, cell data and field data)
# mrichard@esrf.fr & maxime.dupraz@esrf.fr
########################################################################################################################
########################################################################################################################

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



#'/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'
savefile = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_3/'
#'/home/maxime/Documents/maxime/Reseau/maxime/Beamtimes/P10/Facet_analyser/results/'
#'/data/id01/inhouse/maxime/Beamtimes/P10/Facet_analyser/results/'
comment = ''
name = 'facet'
iso = 0.5
ref_normal = [1,1,1]/norm([1,1,1])
ref_string = ' '.join(str('{:.0f}'.format(e)) for e in ref_normal)
comment = comment + '_ref_' + ref_string
pathload = '/home/dupraz/Documents/Post_doc_ESRF/Beamtimes/P-10_05_2019/facet_analyzer/results/crystal_3/align_02_%05d/'
#'/home/maxime/Documents/maxime/Reseau/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'
tol = 0.164 # by default 0.08
#'/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'

# scan numbers -- crystal "small" P10
scans = [787,805,824,828,839,842,856,859,1175,1179,1183,1199,1212,1244] 
#[890,905,910,922,926,946,950,966,970,986,
#990,994,1010,1022,1034,1047,1059,1065,1081,1085,1100]  
nb_facets = 100

for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_facets = min(nb_facets,len(data['facet']))
    print('Number of facets for scan %s is %s' % (scans[ii], len(data['facet'])))

print('Min. number of facets:', nb_facets)

# remove scan 1010 (too low number of identified facet)
#scans = [890,905,910,922,926,946,950,966,970,986,
#         990,994,1022,1034]  

nb_facets_min, nb_facets_max = 100, 0
for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_facets_min = min(nb_facets_min,len(data['facet']))
    nb_facets_max = max(nb_facets_max,len(data['facet']))
    print('Number of facets for scan %s is %s' % (scans[ii], len(data['facet'])))

print('Min. number of facets:', nb_facets_min)
print('Max. number of facets:', nb_facets_max)

# =============================================================================
# Plot average facet displacement & strain as a function of scan
# =============================================================================
x = []
y_disp = []
y_strain = []

for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_Facet = len(data['facet'])
    for jj in range(nb_Facet):
        x.append(ii) 
        y_disp.append(data['disp_mean'][jj])
        y_strain.append(data['strain_mean'][jj])

#plt.figure()
#plt.subplot(2,1,1)
#plt.scatter(x,y_disp)
#plt.xlabel('scan')
#plt.ylabel('average facet displacement')
#plt.subplot(2,1,2)
#plt.scatter(x,y_strain,vmin=-0.0005,vmax=0.0005)
#plt.axis([-1,14,-0.0005,0.0005])
#plt.xlabel('scan')
#plt.ylabel('average facet strain')
#plt.show()

# =============================================================================
# rotation matrix
# =============================================================================
u0 = [1,1,1]/np.sqrt(3)
v0 = [1,1,-2]/np.sqrt(6)
w0 = [-1,1,0]/np.sqrt(2)

np.cross(u0,v0) # compares well with w0


u = [-0.326415,	0.926122,	0.0615047] # [111]
w = [0.607296,	0.244736,	0.74306]  # [-1,1,0] 
v =np.cross(w/np.linalg.norm(w),u/np.linalg.norm(u)) # [1,1,-2] 


u1 = u/np.linalg.norm(u)
v1 = v/np.linalg.norm(v)
w1 = w/np.linalg.norm(w)

# matrice changement de base
a = np.array([u0,v0,w0])
b = np.array([u1,v1,w1])
invb = np.linalg.inv(b)
M_rot = np.dot(np.transpose(a),np.transpose(invb))

# =============================================================================
# Print normals
# =============================================================================
ii = 0
data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
nb_Facet = len(data['facet'])
for jj in range(nb_Facet):
    normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
    normals = np.dot(M_rot,normals)
    print(normals)
    
# =============================================================================
# Find the strain / displacement for each facet by comparing the vector normal 
# to the facet
# =============================================================================
data_ini = np.load(pathload%scans[0]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
nb_Facet_ini = len(data_ini['facet'])



res = {}
res['facet_nb'] = np.zeros(nb_Facet_ini)
res['strain'] = np.zeros((nb_Facet_ini,len(scans)))
res['strain_std'] = np.zeros((nb_Facet_ini,len(scans)))
res['disp'] = np.zeros((nb_Facet_ini,len(scans)))
res['disp_std'] = np.zeros((nb_Facet_ini,len(scans)))
res['scan'] = np.zeros((nb_Facet_ini,len(scans)))
res['real_facet_nb'] = np.zeros((nb_Facet_ini,len(scans)))
res['condition'] = np.zeros((nb_Facet_ini,len(scans)))
res['normal'] = np.zeros((nb_Facet_ini,len(scans),3))

for jj in range(nb_Facet_ini):
    res['facet_nb'][jj] = jj
    normals = np.array([data_ini['n0'][jj],data_ini['n1'][jj],data_ini['n2'][jj]])
    normals = np.dot(M_rot,normals)
    print(normals)
    ind_ = 0
    res['strain'][jj,ind_] = data_ini['strain_mean'][jj]
    res['strain_std'][jj,ind_] = data_ini['strain_std'][jj]
    res['disp'][jj,ind_] = data_ini['disp_mean'][jj]
    res['disp_std'][jj,ind_] = data_ini['disp_std'][jj]    
    res['scan'][jj,ind_] = scans[ind_]    
    res['real_facet_nb'][jj,ind_] = jj    
    res['condition'][jj,ind_] = ind_ 
    res['normal'][jj,ind_,:] = normals    
    for ii in range(1,len(scans)):
        # open data files other than the initial one
        data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
        # check if one norm coincides with normals
        for tt in range(len(data['facet'])):
            norm_ = np.array([data['n0'][tt],data['n1'][tt],data['n2'][tt]])
            norm_ = np.dot(M_rot,norm_)
            # calculate the difference for the 3 components of the norm
            diff0, diff1, diff2 = abs(normals[0]-norm_[0]),abs(normals[1]-norm_[1]),abs(normals[2]-norm_[2])
            if ((diff0 < tol) and (diff1 < tol) and (diff2 < tol)):
                ind_ = ind_ + 1
                res['strain'][jj,ind_] = data['strain_mean'][tt]
                res['scan'][jj,ind_] = scans[ii]    
                res['real_facet_nb'][jj,ind_] = tt  
                res['condition'][jj,ind_] = ii
                res['normal'][jj,ind_,:] = norm_ 
                res['strain_std'][jj,ind_] = data['strain_std'][tt]
                res['disp'][jj,ind_] = data['disp_mean'][tt]
                res['disp_std'][jj,ind_] = data['disp_std'][tt]   

#plt.figure()
#plt.errorbar(res['condition'][16],res['strain'][16],res['strain_std'][16], fmt='o',capsize=2)
#plt.errorbar(res['condition'][30],res['strain'][30],res['strain_std'][30], fmt='o',capsize=2)
#plt.ylabel('Retrieved <strain>')
#plt.xlabel('Gas condition')
#plt.grid()  
#plt.show() 

# =============================================================================
# Labelling facets by hands
# =============================================================================

#Ref scan 787


hkl_facets = [[0, -1, 1],[2, -2, 1],[1, -1, 1],[0, -1, 0],[-1, -3, 1],[1, -3, -1],
[1, -1, 0],[1, -1, 3],[3, -1, 1],[-1, -1, 1] ,[-1, -1, 3],[1, 0, 1],[0, 0, 1],
[1, -1, -1],[3, -1, -1],[1, 0, 0],[1, 1, 3],[3, 1, 1],[-3, -1, 1],[-1, 0, 1],
[-1, 1, 3],[-1, -1, -1],[1, 1, 1],[1, -1, -3],[1, 0, -1],[3, 1, -1],
[-1, 0, 0],[0, 1, 1],[-3, 1, 1],[-1, 1, 1],[1, 1, 0],[0, 0, -1],[1, 1, -3],
[1, 1, -1],[1, 3, 1],[-3, 1, -1],[-1, 1, 0],[-1, 3, 1],[-1, 1, -3],[1, 3, -1],
[0, 1, 0],[0, 1, -1],[-1, 1, -1],[-1, 3, -1]]



# =============================================================================
# Plot with legend
# =============================================================================
legend = []

#for ii in range(nb_Facet_ini):
for ii in range(nb_Facet_ini):    
    legend = legend + [' '.join(str('{:.0f}'.format(e)) for e in hkl_facets[ii])]


plt.figure(figsize=(15,10))
for ii in range(nb_Facet_ini):
    plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
    #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    plt.savefig(savefile+ name + '_strain.png')
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.show()
# =============================================================================
# Plot depending on the plane family
# =============================================================================


plt.figure(figsize=(15,15))
plt.subplot(4,1,1)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,2)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,3)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=10, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,4)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=1,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=10, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.savefig(savefile+ name + '_strain_organized.png')
plt.show()



plt.figure(figsize=(15,12))
plt.subplot(4,1,1)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,2)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,3)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=10, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,4)
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=1,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=10, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.savefig(savefile+ name + '_disp_organized.png')
plt.show()

# =============================================================================
# Create output file
# =============================================================================

# create data file
for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','disp','disp_std','strain','strain_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
        f.close()


for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','disp','disp_std','strain','strain_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()


for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','disp','disp_std','strain','strain_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()


for ii in range(nb_Facet_ini):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','disp','disp_std','strain','strain_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()










# =============================================================================
# Common facets
# =============================================================================
#new_angle = np.zeros((len(scans),len(hkl_facets)))
#ref_normal = [1,1,1]/norm([1,1,1])
#
#for ii in range(len(scans)):
#    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
#    nb_Facet = len(data['facet'])
#    for jj in range(nb_Facet):
#        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
#        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
#        if (data['n2'][jj] < 0):
#            new_angle[ii,jj] = - new_angle[ii,jj]
#
#
#fig = plt.figure()
#plt.plot(new_angle[0:2,:].T,'-o')
#plt.plot(new_angle[6,:].T,'-o')
#plt.show()
#
## remove facets
#new_angle = np.zeros((len(scans),len(hkl_facets)))
#ref_normal = [1,1,1]/norm([1,1,1])
#
#for ii in range(len(scans)):
#    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
#    nb_Facet = len(data['facet'])
#    for jj in range(nb_Facet):
#        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
#        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
#        if (data['n2'][jj] < 0):
#            new_angle[ii,jj] = - new_angle[ii,jj]
#
#
#fig = plt.figure()
#plt.plot(new_angle[0:2,:].T,'-o')
#plt.plot(new_angle[6,:].T,'-o')
#plt.show()
#
#
#
#
## =============================================================================
## Classification of facets (normals)
## =============================================================================
#new_angle = np.zeros((len(scans),len(hkl_facets)))
#ref_normal = [1,1,1]/norm([1,1,1])
#
#for ii in range(len(scans)):
#    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
#    nb_Facet = len(data['facet'])
#    for jj in range(nb_Facet):
#        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
#        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
#        if (data['n2'][jj] < 0):
#            new_angle[ii,jj] = - new_angle[ii,jj]
#
#
#fig = plt.figure()
#plt.plot(new_angle[0:3,:].T,'-o')
#plt.show()





