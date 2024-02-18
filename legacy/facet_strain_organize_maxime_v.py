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
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import os
import vtk


pathfile = '/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'
savefile = '/data/id01/inhouse/maxime/Beamtimes/P10/Facet_analyser/results/'
comment = ''
name = 'facet'
iso = 0.5
ref_normal = [1,1,1]/norm([1,1,1])
ref_string = ' '.join(str('{:.0f}'.format(e)) for e in ref_normal)
comment = comment + '_ref_' + ref_string
pathload = '/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'

# scan numbers -- crystal "small" P10
scans = [890,905,910,922,926,946,950,966,970,986,
         990,994,1022,1034,1047,1059,1065,1081,1085,1100]  
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

plt.figure()
plt.subplot(2,1,1)
plt.scatter(x,y_disp)
plt.xlabel('scan')
plt.ylabel('average facet displacement')
plt.subplot(2,1,2)
plt.scatter(x,y_strain,vmin=-0.0005,vmax=0.0005)
plt.axis([-1,14,-0.0005,0.0005])
plt.xlabel('scan')
plt.ylabel('average facet strain')

# =============================================================================
# rotation matrix
# =============================================================================
u0 = [1,1,1]/np.sqrt(3)
v0 = [-1,-1,2]/np.sqrt(6)
w0 = [1,-1,0]/np.sqrt(2)

np.cross(u0,v0) # compares well with w0


u = [0.0098667107667,0.99203637508,0.042553117068] # [111]
# v = [0.695623810186964,0,0.78033019452292] # [-1,-1,2] // [0,0,1]
v = [0.63884773559,1-0.99203637508,-0.47306533899] # [-1,-1,2] // [0,0,1]
w = np.cross(u/np.linalg.norm(u),v/np.linalg.norm(v))

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
#nb_Facet_ini = len(data_ini['facet'])

hkl_facets = [[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[1,0,0],[0,-1,0],
              [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,-1,1],
              [0,-1,1],[0,1,-1],[1,0,-1],[-1,0,1],[0,-1,1],[0,1,-1],
              [1,0,1],[-1,0,-1],[0,1,1],[0,-1,-1],[1,1,0],[-1,-1,0],
              [-1,-1,3],[-1,1,3],[1,-1,-3],[1,1,-3],[-1,1,-3],[-1,-3,1],
              [1,-3,-1],[1,-3,1],[-1,3,-1],[-3,1,1],[3,-1,1],[3,-1,-1]]


res = {}
res['facet_nb'] = np.zeros(len(hkl_facets))
res['strain'] = np.zeros((len(hkl_facets),len(scans)))
res['strain_std'] = np.zeros((len(hkl_facets),len(scans)))
res['disp'] = np.zeros((len(hkl_facets),len(scans)))
res['disp_std'] = np.zeros((len(hkl_facets),len(scans)))
res['scan'] = np.zeros((len(hkl_facets),len(scans)))
res['real_facet_nb'] = np.zeros((len(hkl_facets),len(scans)))
res['condition'] = np.zeros((len(hkl_facets),len(scans)))
res['normal'] = np.zeros((len(hkl_facets),len(scans),3))

for jj in range(len(hkl_facets)):
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
            if ((diff0 < 0.08) and (diff1 < 0.08) and (diff2 < 0.08)):
                ind_ = ind_ + 1
                res['strain'][jj,ind_] = data['strain_mean'][tt]
                res['scan'][jj,ind_] = scans[ii]    
                res['real_facet_nb'][jj,ind_] = tt  
                res['condition'][jj,ind_] = ii
                res['normal'][jj,ind_,:] = norm_ 
                res['strain_std'][jj,ind_] = data['strain_std'][tt]
                res['disp'][jj,ind_] = data['disp_mean'][tt]
                res['disp_std'][jj,ind_] = data['disp_std'][tt]   

plt.figure()
plt.errorbar(res['condition'][16],res['strain'][16],res['strain_std'][16], fmt='o',capsize=2)
plt.errorbar(res['condition'][30],res['strain'][30],res['strain_std'][30], fmt='o',capsize=2)
plt.ylabel('Retrieved <strain>')
plt.xlabel('Gas condition')
plt.grid()   

# =============================================================================
# Labelling facets by hands
# =============================================================================
#hkl_facets = [[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[1,0,0],[0,-1,0],
#              [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,-1,1],
#              [0,-1,1],[0,1,-1],[1,0,-1],[-1,0,1],[0,-1,1],[0,1,-1],
#              [1,0,1],[-1,0,-1],[0,1,1],[0,-1,-1],[1,1,0],[-1,-1,0],
#              [-1,-1,3],[-1,1,3],[1,-1,-3],[1,1,-3],[-1,1,-3],[-1,-3,1],
#              [1,-3,-1],[1,-3,1],[-1,3,-1],[-3,1,1],[3,-1,1],[3,-1,-1]]

#[[1,-3,1],[0,-1,1],[1,-1,1],[0,-1,0],[-1,-3,1],[1,-1,0],[1,-3,-1],[-1,-1,1],[-1,-1,3],[0,0,1],[1,-1,-1],[1,0,1],[1,0,0],[-5,-1,1],[-1,0,1],[-1,1,3],[-1,-1,-1],[1,-1,-3],[1,1,1],[1,0,-1],[0,1,1],[-1,0,0],[1,1,0],[-3,1,1],[-1,1,1],[0,0,-1],[-1,0,-1],[1,1,-3],[1,1,-1],[-1,1,0],[0,1,0],[-1,0,-1],[0,1,-1]]

# =============================================================================
# Plot with legend
# =============================================================================
legend = []

#for ii in range(nb_Facet_ini):
for ii in range(len(hkl_facets)):    
    legend = legend + [' '.join(str('{:.0f}'.format(e)) for e in hkl_facets[ii])]


plt.figure(figsize=(10,6))
for ii in range(len(hkl_facets)):
    plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
    #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()

# =============================================================================
# Plot depending on the plane family
# =============================================================================


plt.figure(figsize=(7,7))
plt.subplot(4,1,1)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,2)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,3)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,4)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        plt.errorbar(res['condition'][ii],res['strain'][ii],res['strain_std'][ii], fmt='-o',capsize=1,label=legend[ii])
        #plt.xlabel('Angle (deg.)')6
    plt.ylabel('Retrieved <strain>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()



plt.figure(figsize=(7,7))
plt.subplot(4,1,1)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=8, fancybox=True, shadow=True)
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,2)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,3)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=2,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()
plt.subplot(4,1,4)
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        plt.errorbar(res['condition'][ii],res['disp'][ii],res['disp_std'][ii], fmt='o',capsize=1,label=legend[ii])
        #plt.xlabel('Angle (deg.)')
    plt.ylabel('Retrieved <displacement>')
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
    plt.grid()

# =============================================================================
# Create output file
# =============================================================================

# create data file
for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,1])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','strain','strain_std','disp','disp_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
        f.close()


for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,0,0])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','strain','strain_std','disp','disp_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()


for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,0])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','strain','strain_std','disp','disp_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()


for ii in range(len(hkl_facets)):
    if (norm(hkl_facets[ii])==norm([1,1,3])):
        f = open(savefile+'strain_disp_'+legend[ii]+'.dat',"w+")
        f.write("%s %s %s %s %s %s\r\n" %('condition','strain','strain_std','disp','disp_std','hkl'))
        nb = len(res['condition'][ii])
        for jj in range(nb):   
            f.write("%f %f %f %f %f %s\r\n" % (res['condition'][ii,jj],res['disp'][ii,jj],res['disp_std'][ii,jj],res['strain'][ii,jj],res['strain_std'][ii,jj],'('+legend[ii]+')'))
f.close()










# =============================================================================
# Common facets
# =============================================================================
new_angle = np.zeros((len(scans),len(hkl_facets)))
ref_normal = [1,1,1]/norm([1,1,1])

for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_Facet = len(data['facet'])
    for jj in range(nb_Facet):
        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
        if (data['n2'][jj] < 0):
            new_angle[ii,jj] = - new_angle[ii,jj]


fig = plt.figure()
plt.plot(new_angle[0:2,:].T,'-o')
plt.plot(new_angle[6,:].T,'-o')
plt.show()

# remove facets
new_angle = np.zeros((len(scans),len(hkl_facets)))
ref_normal = [1,1,1]/norm([1,1,1])

for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_Facet = len(data['facet'])
    for jj in range(nb_Facet):
        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
        if (data['n2'][jj] < 0):
            new_angle[ii,jj] = - new_angle[ii,jj]


fig = plt.figure()
plt.plot(new_angle[0:2,:].T,'-o')
plt.plot(new_angle[6,:].T,'-o')
plt.show()




# =============================================================================
# Classification of facets (normals)
# =============================================================================
new_angle = np.zeros((len(scans),len(hkl_facets)))
ref_normal = [1,1,1]/norm([1,1,1])

for ii in range(len(scans)):
    data = np.load(pathload%scans[ii]+'name'+'_iso='+str(iso) + comment +'.npy',allow_pickle='False').item()
    nb_Facet = len(data['facet'])
    for jj in range(nb_Facet):
        normals = np.array([data['n0'][jj],data['n1'][jj],data['n2'][jj]])
        new_angle[ii,jj] = math.acos(np.dot(ref_normal,normals/norm(normals)))*180./np.pi
        if (data['n2'][jj] < 0):
            new_angle[ii,jj] = - new_angle[ii,jj]


fig = plt.figure()
plt.plot(new_angle[0:3,:].T,'-o')
plt.show()


## =============================================================================
## Options
## =============================================================================
#rotate_particle = True
#fixed_reference = True
#
## =============================================================================
## rotation matrix
## =============================================================================
#u0 = [1,1,1]/np.sqrt(3)
#v0 = [-1,-1,2]/np.sqrt(6)
#w0 = [1,-1,0]/np.sqrt(2)
#
#np.cross(u0,v0) # compares well with w0
#
#
#u = [0.0098667107667,0.99203637508,0.042553117068] # [111]
## v = [0.695623810186964,0,0.78033019452292] # [-1,-1,2] // [0,0,1]
#v = [0.63884773559,1-0.99203637508,-0.47306533899] # [-1,-1,2] // [0,0,1]
#w = np.cross(u/np.linalg.norm(u),v/np.linalg.norm(v))
#
#u1 = u/np.linalg.norm(u)
#v1 = v/np.linalg.norm(v)
#w1 = w/np.linalg.norm(w)
#
## matrice changement de base
#a = np.array([u0,v0,w0])
#b = np.array([u1,v1,w1])
#invb = np.linalg.inv(b)
#M_rot = np.dot(np.transpose(a),np.transpose(invb))
#
#dir111 = np.array([0.5597,0.5241,-0.64188]) #[100]
#print(np.dot(M_rot,dir111/np.linalg.norm(dir111)))



## =============================================================================
## Load VTK file
## =============================================================================
#
#scan = 1034
#name = 'facet'
#
#pathdir = '/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/'
#pathsave = '/data/id01/inhouse/maxime/Beamtimes/P10/Facet_analyser/results/align_02_%05d/'
##'/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/facet_strain/'
#
#if not os.path.exists(pathsave%scan):
#    os.makedirs(pathsave%scan)
#
#filename, iso = pathdir %scan + name + '.vtk', 0.5
#
#
#reader = vtk.vtkGenericDataObjectReader()
#reader.SetFileName(filename)
#reader.ReadAllScalarsOn()
#reader.ReadAllVectorsOn()
#reader.ReadAllTensorsOn()
#reader.Update()
#vtkdata = reader.GetOutput()
#
## =============================================================================
## Get point data
## =============================================================================
#
#
#pointData = vtkdata.GetPointData()
#print("Number of points = %s" % str(vtkdata.GetNumberOfPoints()))
#print("Number of cells = %s" % str(vtkdata.GetNumberOfCells()))
#
#input = {}
#input['disp'] = np.zeros(vtkdata.GetNumberOfPoints())
#input['strain'] = np.zeros(vtkdata.GetNumberOfPoints())
#input['x'] = np.zeros(vtkdata.GetNumberOfPoints())
#input['y'] = np.zeros(vtkdata.GetNumberOfPoints())
#input['z'] = np.zeros(vtkdata.GetNumberOfPoints())
#
## pointData.GetArrayName(1) # to get the name of the array
## get the positions of the points-voxels // vtkdata.GetPoint(0)[0] or vtkdata.GetPoint(0)
#for ii in range(vtkdata.GetNumberOfPoints()):
#    input['x'][ii] = vtkdata.GetPoint(ii)[0]
#    input['y'][ii] = vtkdata.GetPoint(ii)[1]
#    input['z'][ii] = vtkdata.GetPoint(ii)[2]
#    input['strain'][ii] = pointData.GetArray('strain').GetValue(ii)
#    input['disp'][ii] = pointData.GetArray('disp').GetValue(ii)
#
#
## =============================================================================
## Get cell data
## =============================================================================
#    
#cellData = vtkdata.GetCellData()
#input['FacetProbabilities'] = np.zeros(vtkdata.GetNumberOfCells())
#input['FacetIds'] = np.zeros(vtkdata.GetNumberOfCells())
#input['x0'] = np.zeros(vtkdata.GetNumberOfCells())
#input['y0'] = np.zeros(vtkdata.GetNumberOfCells())
#input['z0'] = np.zeros(vtkdata.GetNumberOfCells())
#
#for ii in range(vtkdata.GetNumberOfCells()):
#    input['FacetProbabilities'][ii] = cellData.GetArray('FacetProbabilities').GetValue(ii)
#    input['FacetIds'][ii] = cellData.GetArray('FacetIds').GetValue(ii)
#    input['x0'][ii] = vtkdata.GetCell(ii).GetPointId(0)
#    input['y0'][ii] = vtkdata.GetCell(ii).GetPointId(1)
#    input['z0'][ii] = vtkdata.GetCell(ii).GetPointId(2)
#
#nb_Facet = int(max(input['FacetIds']))
#print("Number of facets = %s" % str(nb_Facet))
#
#
#def extract_facet(facet_id,input):
#    ind_Facet = []
#    for ii in range(len(input['FacetIds'])):
#        if (int(input['FacetIds'][ii]) == facet_id):
#            ind_Facet.append(input['x0'][ii])
#            ind_Facet.append(input['y0'][ii])
#            ind_Facet.append(input['z0'][ii])
#
#    ind_Facet_new = list(set(ind_Facet))
#    results = {}
#    results['x'], results['y'], results['z'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))
#    results['strain'], results['disp'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))
#
#    for jj in range(len(ind_Facet_new)):
#        results['x'][jj] = input['x'][int(ind_Facet_new[jj])]
#        results['y'][jj] = input['y'][int(ind_Facet_new[jj])]
#        results['z'][jj] = input['z'][int(ind_Facet_new[jj])]
#        results['strain'][jj] = input['strain'][int(ind_Facet_new[jj])]
#        results['disp'][jj] = input['disp'][int(ind_Facet_new[jj])]
#    results['strain_mean'] = np.mean(results['strain'])
#    results['strain_std'] = np.std(results['strain'])
#    results['disp_mean'] = np.mean(results['disp'])
#    results['disp_std'] = np.std(results['disp'])
#    return results
#
#
#
## plot single result
#result = extract_facet(12,input)
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(input['x'], input['y'], input['z'], s=0.2, antialiased=True, depthshade=True)
#ax.scatter(result['x'], result['y'], result['z'],s=50, c = result['strain'], cmap = 'jet',  vmin = -0.025, vmax = 0.025, antialiased=True, depthshade=True)
#plt.show()
#
## =============================================================================
## # plot 3D strain and displacement 
## =============================================================================
#
## Displacement
#
#fig = plt.figure(figsize=((10,10)))
#ax = fig.gca(projection='3d')
##ax.scatter(input['x'], input['y'], input['z'], s=0.2, antialiased=True, depthshade=True)
#for tt in np.arange(1, nb_Facet+1, 1):
#    print("Facet = %d" % tt)
#    results = extract_facet(tt,input)
#    ax.scatter(results['x'], results['y'],results['z'], s=50, c = results['disp']
#    , cmap = 'jet',  vmin = -disp_range, vmax = disp_range, antialiased=True, depthshade=True)
##    colorbar = fig.colorbar(ax.scatter, ax=ax)
#ax.view_init(elev=-64, azim=94)
#plt.show()
#plt.savefig(pathsave%scan+ name + '_disp_3D_' + hkls + '_' +str(disp_range) +'.png')
#
## Strain
#
#fig = plt.figure(figsize=(10,10))
#ax = fig.gca(projection='3d')
##ax.scatter(input['x'], input['y'], input['z'], s=0.2, antialiased=True, depthshade=True)
#for tt in np.arange(1, nb_Facet+1, 1):
#    print("Facet = %d" % tt)
#    results = extract_facet(tt,input)
#    ax.scatter(results['x'], results['y'],results['z'], s=50, c = results['strain']
#    , cmap = 'jet',  vmin = -strain_range, vmax = strain_range, antialiased=True, depthshade=True)
##    colorbar = fig.colorbar(ax.scatter, ax=ax)
#ax.view_init(elev=-64, azim=94)
#plt.show()
#plt.savefig(pathdir%scan+ name + '_strain_3D_' + hkls + '_' +str(strain_range) +'.png')
#
#
## =============================================================================
## # Extract strain and displacement parameters
## =============================================================================
#
#facet = np.arange(1, int(nb_Facet) + 1, 1)
#strain_mean = np.zeros(len(facet))
#strain_std = np.zeros(len(facet))
#disp_mean = np.zeros(len(facet))
#disp_std = np.zeros(len(facet))
#strain_mean_facets=[]
#disp_mean_facets=[]
#
#for tt in np.arange(1, int(nb_Facet) + 1, 1):
#    print("Facet = %d" % tt)
#    results = extract_facet(tt,input)
#    strain_mean[tt-1] = results['strain_mean']
#    strain_std[tt-1] = results['strain_std']
#    disp_mean[tt-1] = results['disp_mean']
#    disp_std[tt-1] = results['disp_std']
#
## =============================================================================
## # plot average 3D strain and displacement
## =============================================================================
#
#
## Average disp
#
#fig = plt.figure(figsize=(10,10))
#ax = fig.gca(projection='3d')
##ax.scatter(input['x'], input['y'], input['z'], s=0.2, antialiased=True, depthshade=True)
#for tt in np.arange(1, nb_Facet+1, 1):
#    print("Facet = %d" % tt)
#    results = extract_facet(tt,input)
#    disp_mean_facet = np.zeros(results['disp'].shape)
#    disp_mean_facet.fill(results['disp_mean'])
#    disp_mean_facets=np.append(strain_mean_facets,disp_mean_facet,axis=0)
#    ax.scatter(results['x'], results['y'],results['z'], s=50, c = disp_mean_facet
#    , cmap = 'jet',  vmin = -disp_range_avg/2, vmax = disp_range_avg/2, antialiased=True, depthshade=True)
##    colorbar = fig.colorbar(ax.scatter, ax=ax)
#ax.view_init(elev=-64, azim=94)
#plt.show()
#plt.savefig(pathsave%scan+ name + '_disp_3D_avg_' + hkls + '_' +str(disp_range_avg) +'.png')
#
## Average strain    
#    
#fig = plt.figure(figsize=(10,10))
#ax = fig.gca(projection='3d')
##ax.scatter(input['x'], input['y'], input['z'], s=0.2, antialiased=True, depthshade=True)
#for tt in np.arange(1, nb_Facet+1, 1):
#    print("Facet = %d" % tt)
#    results = extract_facet(tt,input)
#    strain_mean_facet = np.zeros(results['strain'].shape)
#    strain_mean_facet.fill(results['strain_mean'])
#    strain_mean_facets=np.append(strain_mean_facets,strain_mean_facet,axis=0)
#    ax.scatter(results['x'], results['y'],results['z'], s=50, c = strain_mean_facet
#    , cmap = 'jet',  vmin = -strain_range_avg, vmax = strain_range_avg, antialiased=True, depthshade=True)
##    colorbar = fig.colorbar(ax.scatter, ax=ax)
#ax.view_init(elev=-64, azim=94)
#plt.show()
#plt.savefig(pathsave%scan+ name + '_strain_3D_avg_' + hkls + '_' +str(strain_range_avg) +'.png')
#
## =============================================================================
## # Get field data
## =============================================================================
#field = {}
#fieldData = vtkdata.GetFieldData()
#
#field['facet'] = facet
#field['strain_mean'] = strain_mean
#field['strain_std'] = strain_std
#field['disp_mean'] = disp_mean
#field['disp_std'] = disp_std
#
#field['n0'] = np.zeros(nb_Facet)
#field['n1'] = np.zeros(nb_Facet)
#field['n2'] = np.zeros(nb_Facet)
#field['FacetIds'] = np.zeros(nb_Facet)
#field['absFacetSize'] = np.zeros(nb_Facet)
#field['interplanarAngles'] = np.zeros(nb_Facet)
#field['relFacetSize'] = np.zeros(nb_Facet)
#
#
#
#for ii in range(nb_Facet):
#    field['n0'][ii] = fieldData.GetArray('facetNormals').GetValue(3*ii)
#    field['n1'][ii] = fieldData.GetArray('facetNormals').GetValue(3*ii+1)
#    field['n2'][ii] = fieldData.GetArray('facetNormals').GetValue(3*ii+2)
#    field['FacetIds'][ii] = fieldData.GetArray('FacetIds').GetValue(ii)
#    field['absFacetSize'][ii] = fieldData.GetArray('absFacetSize').GetValue(ii)
#    field['interplanarAngles'][ii] = fieldData.GetArray('interplanarAngles').GetValue(ii)
#    field['relFacetSize'][ii] = fieldData.GetArray('relFacetSize').GetValue(ii)
#
## =============================================================================
## # Get normals
## =============================================================================
#
#A = np.zeros(1)
#    
#normals = np.zeros((nb_Facet,3))
#field['interplanarAngles']=np.delete(field['interplanarAngles'],nb_Facet-1,0)
#field['interplanarAngles']=np.append(A,field['interplanarAngles'],axis=0)
#
#
#for ii in range(nb_Facet):
#    normals[ii]= np.array([field['n0'][ii],field['n1'][ii],field['n2'][ii]])
#    
## =============================================================================
## # Rotate particle if required
## =============================================================================
#
#if rotate_particle:
#    for ii in range(nb_Facet):
#        normals[ii] = np.dot(M_rot,normals[ii])
#
## =============================================================================
## # Interplanar angle fixed reference
## =============================================================================
#
#if fixed_reference:
#    for ii in range(nb_Facet):
#        field['interplanarAngles'][ii] = math.acos(np.dot(ref_normal,normals[ii]/norm(normals[ii])))*180./np.pi
#
#    comment = comment + '_ref_' + ref_string
#
## =============================================================================
## # Prepare the legend for the 1D plots
## =============================================================================
#
#
#legend = []
#
#for ii in range(nb_Facet):
#    #legend = legend + [' '.join(str('{:.1f}'.format(e)) for e in normals[ii])]
#    #legend = legend + [' '.join(str('{:.0f}'.format(e)) for e in normals[ii]*2)]
#    legend = legend + [' '.join(str('{:.2f}'.format(e)) for e in normals[ii])]
#
## =============================================================================
## # 1D plot: average displacement vs facet index
## =============================================================================
#plt.figure(figsize=(10,6))
#for ii in range(nb_Facet):
#    plt.errorbar(facet[ii], disp_mean[ii], disp_std[ii], fmt='o',label=legend[ii])
#    plt.xlabel('Facet index')
#    plt.ylabel('Average retrieved displacement')
#    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
#    plt.grid()
#plt.show()
#plt.savefig(pathsave%scan+ name + '_avg_disp_vs_facet_id_' + hkls + comment +'.png')
#
## =============================================================================
## # 1D plot: average strain vs facet index
## =============================================================================
#
#plt.figure(figsize=(10,6))
#for ii in range(nb_Facet):
#    plt.errorbar(facet[ii], strain_mean[ii], strain_std[ii], fmt='o',label=legend[ii])
#    plt.xlabel('Facet index')
#    plt.ylabel('Average retrieved strain')
#    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
#    plt.grid()
#plt.show()
#plt.savefig(pathsave%scan+ name + '_avg_strain_vs_facet_id_' + hkls + comment +'.png')
#
## =============================================================================
## # 1D plot: average strain vs angle with respect to the reference facet
## # 1D plot: average displacement vs angle with respect to the reference facet
## # 1D plot: relative facet size vs angle with respect to the reference facet
#
## =============================================================================
#    
#plt.figure(figsize=(10,12))
#plt.subplot(3,1,1)
#for ii in range(nb_Facet):
#    plt.errorbar(field['interplanarAngles'][ii], disp_mean[ii], disp_std[ii], fmt='o',capsize=2,label=legend[ii])
#    #plt.xlabel('Angle (deg.)')6
#    plt.ylabel('Retrieved <disp> (Angstroms)')
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=5, fancybox=True, shadow=True)
#    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
#    plt.grid()
#plt.subplot(3,1,2)
#for ii in range(nb_Facet):
#    plt.errorbar(field['interplanarAngles'][ii], strain_mean[ii], strain_std[ii], fmt='o',capsize=2,label=legend[ii])
#    #plt.xlabel('Angle (deg.)')
#    plt.ylabel('Retrieved <strain>')
#    plt.grid()
#plt.subplot(3,1,3)
#for ii in range(nb_Facet):
#    plt.plot(field['interplanarAngles'][ii], field['relFacetSize'][ii],'o',label=legend[ii])
#    plt.xlabel('Angle (deg.)')
#    plt.ylabel('Relative facet size')
#    plt.grid()
#    plt.show()
#plt.savefig(pathsave%scan+ name + '_disp_strain_size_vs_angle_planes_' + hkls + comment + '.png')
#
#plt.figure(figsize=(10,12))
#plt.subplot(3,1,1)
#for ii in range(nb_Facet):
#    lx,ly,lz = float(legend[ii].split()[0]),float(legend[ii].split()[1]),float(legend[ii].split()[2])
#    if (lx>0 and ly>0):
#        fmt='o'
#    if (lx>0 and ly<0):
#        fmt='d'
#    if (lx<0 and ly>0):
#        fmt='s'
#    if (lx<0 and ly<0):
#        fmt = '+'
#    plt.errorbar(field['interplanarAngles'][ii], disp_mean[ii], disp_std[ii], fmt=fmt,capsize=2,label=legend[ii])
#    #plt.xlabel('Angle (deg.)')6
#    plt.ylabel('Retrieved <disp> (Angstroms)')
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=5, fancybox=True, shadow=True)
#    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
#    plt.grid()
#plt.subplot(3,1,2)
#for ii in range(nb_Facet):
#    lx,ly,lz = float(legend[ii].split()[0]),float(legend[ii].split()[1]),float(legend[ii].split()[2])
#    if (lx>0 and ly>0):
#        fmt='o'
#    if (lx>0 and ly<0):
#        fmt='d'
#    if (lx<0 and ly>0):
#        fmt='s'
#    if (lx<0 and ly<0):
#        fmt = '+'
#    plt.errorbar(field['interplanarAngles'][ii], strain_mean[ii], strain_std[ii], fmt=fmt,capsize=2,label=legend[ii])
#    #plt.xlabel('Angle (deg.)')
#    plt.ylabel('Retrieved <strain>')
#    plt.grid()
#plt.subplot(3,1,3)
#for ii in range(nb_Facet):
#    plt.plot(field['interplanarAngles'][ii], field['relFacetSize'][ii],'o',label=legend[ii])
#    plt.xlabel('Angle (deg.)')
#    plt.ylabel('Relative facet size')
#    plt.grid()
#    plt.show()
#plt.savefig(pathsave%scan+ name + '_disp_strain_size_vs_angle_planes_' + hkls + comment + '.png')
#
#
#'''
## =============================================================================
## # Plot as a function of hkl families
## =============================================================================
#plt.figure(figsize=(10,12))
#plt.subplot(1,1,1)
#for ii in range(nb_Facet):
#    lx,ly,lz = int(legend[ii].split()[0]),int(legend[ii].split()[1]),int(legend[ii].split()[2])
#    if ((lx**2+ly**2+lz**2)==3):
#        fmt='o'
#    if ((lx**2+ly**2+lz**2)==5):
#        fmt='d'
#    if ((lx**2+ly**2+lz**2)==6):
#        fmt='s'
#    plt.errorbar(field['interplanarAngles'][ii], disp_mean[ii], disp_std[ii], fmt=fmt,capsize=2,label=legend[ii])
#    #plt.xlabel('Angle (deg.)')6
#    plt.ylabel('Retrieved <disp> (Angstroms)')
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=8, fancybox=True, shadow=True)
#    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
#    plt.grid()
#'''
#
## =============================================================================
## # Save field data
## =============================================================================
#    
##np.save(pathdir+'results_S'+str(scan)+'iso='+str(iso),field)
#np.save(pathsave%scan+'name'+'_iso='+str(iso) + comment,field)
#
#
## create data file
#f = open(pathsave%scan+'name'+'_iso='+str(scan) + comment + ".dat","w+")
#f.write("%s %s %s %s %s %s %s %s %s %s\r\n" % ('facet','interplanarAngles','n1','n2','n3','strain_mean','strain_std','disp_mean','disp_std','relFacetSize'))
#for ii in range(nb_Facet):
#    f.write("%f %f %f %f %f %f %f %f %f %f\r\n" % (ii+1,field['interplanarAngles'][ii],normals[ii][0],normals[ii][1],normals[ii][2],strain_mean[ii],strain_std[ii],disp_mean[ii],disp_std[ii],field['relFacetSize'][ii]))
#f.close()
#
#plt.close("all")




