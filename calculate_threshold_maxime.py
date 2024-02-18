import numpy as np
import h5py


pathfile = '/home/maxime/Documents/maxime/Reseau/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/'
#'/data/id01/inhouse/nili/analysis/Pt/P10/align_02_%05d/pynxraw/modes-HIO-300/'
savefile = '/home/maxime/Documents/maxime/Documents/Post_doc_esrf/Beamtimes/Petra_05_2019/facet_analyzer/results/crystal_3/'

# scan numbers -- crystal "small" P10
# scans = [890,905,910,922,926,946,950,966,970,986,
#         990,994,1010,1022,1034,1047,1059,1065,1081,1085,
#         1100,1105,1126,1130,1159,1167,1171,1259,1286,1290,
#         1306,1312,1324,1335,1355,1376,1416]  

# scan numbers -- crystal 3 P10

#scans = [787,805,824,828,839,845,856,864,1175,1179,1183,1199,1203,1244]  

scans = [787,805,824,828,839,842,856,859,1175,1179,1183,1199,1203,1244]  
threshold=np.zeros(len(scans))

for i in range(len(scans)):
    measure1=h5py.File(pathfile %scans[i]+'modes.h5','r')
    obj1=measure1['entry_1/data_1/data'][0,:]
    amp=np.abs(obj1)
    amp=amp/amp.max()
    threshold[i]=np.mean(amp[amp>0.1])-np.std(amp[amp>0.1])+0.05

print(threshold)

# create file: scan / threshold
data = np.array([scans,threshold])
data = data.T #here you transpose your data, so to have it in two columns

datafile_path = savefile + 'scan-threshold-small-P10.txt'
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt=['%d','%f'])



