
'''
Decode a left-out object category from MVPA data
'''

import sys
from sklearn import svm
import numpy as np
import pdb
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

subj_list=["docnet2001", "docnet2002","docnet2003","docnet2005", "docnet2007","docnet2008", "docnet2012"]

#anatomical ROI
d_roi = ['LOC','PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_distloc',  'APC_distloc', 'PPC_toolloc', 'APC_toolloc']
v_roi = ['LO_toolloc', 'PFS_toolloc']


exp_cond = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5']

exp_cats = ['boat', 'camera',' car', 'guitar','lamp']

#create a list of labels for classification
exp_labels = np.concatenate((np.ones((1,5)),np.ones((1,5))*2, np.ones((1,5))*3, np.ones((1,5))*4, np.ones((1,5))*5),axis =1)[0]

data_dir = 'derivatives/results/beta/catmvpa'

n_vox = 100

#iteratively combine LO and PFS with one of the dorsal ROIS (or on its own)

#do cross-val SVM seperately for each sub
#combine across subs


summary_df = pd.DataFrame(columns = ['sub'] + ["l" + s for s in d_roi] + ["r" + s for s in d_roi])
for sn, ss in enumerate(subj_list):
    subj_dir = f'/lab_data/behrmannlab/vlad/docnet/sub-{ss}/ses-02/{data_dir}'
    roi_dir = f'/lab_data/behrmannlab/vlad/docnet/sub-{ss}/ses-02/derivatives/rois'
    

    roi_decode = []
    for lr in ['l','r']:
        for rr in d_roi:
            
            #load in ROI data for each stim condition
            ventral_data = np.zeros((len(exp_cond), n_vox*2))
            dorsal_data = np.zeros((len(exp_cond), n_vox))
            for ecn, ec in enumerate(exp_cond):
                #load in ventral data
                if os.path.exists(f'{subj_dir}/{lr}LO_toolloc_{ec}.txt'):
                    lo_data = np.loadtxt(f'{subj_dir}/{lr}LO_toolloc_{ec}.txt')
                    if len(lo_data) >= n_vox:
                        lo_data = lo_data[0:n_vox,:]
                else:
                    lo_data = np.zeros((n_vox,4))

                if os.path.exists(f'{subj_dir}/{lr}PFS_toolloc_{ec}.txt'):
                    pfs_data = np.loadtxt(f'{subj_dir}/{lr}PFS_toolloc_{ec}.txt')
                    if len(pfs_data) >= n_vox:
                        pfs_data = pfs_data[0:n_vox,:]

                else:
                    pfs_data = np.zeros((n_vox,4))
                
                
                temp_data = np.append(np.transpose(lo_data[:, 3]),np.transpose(pfs_data[:, 3])) 
                ventral_data[ecn, 0:len(temp_data)] = temp_data 

                #check if dorsal ROI exists and load it in
                if os.path.exists(f'{roi_dir}/{lr}{rr}.nii.gz'):
                    
                    dorsal_roi = np.loadtxt(f'{subj_dir}/{lr}{rr}_{ec}.txt')
                    if len(dorsal_roi) < n_vox:
                        dorsal_data[ecn, 0:len(dorsal_roi)] = np.transpose(dorsal_roi[:, 3])
                    else:
                        dorsal_data[ecn, :] = np.transpose(dorsal_roi[0:n_vox, 3])

            #combine ventral and dorsal roi data
            if rr == 'LOC':
                roi_data = ventral_data
            else:
                roi_data = np.concatenate((ventral_data, dorsal_data), axis = 1)

            #pdb.set_trace()
            #remove zero columns from df
            idx = np.argwhere(np.all(roi_data[..., :] == 0, axis=0))
            roi_data = np.delete(roi_data, idx, axis=1)
            
            #check if ROI exists or is an LOC run before doing SVM
            if os.path.exists(f'{roi_dir}/{lr}{rr}.nii.gz') or rr == 'LOC':
                #run SVM
                X = roi_data
                y = exp_labels
                sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
                sss.get_n_splits(X, y)
                
                roi_acc = []
                for train_index, test_index in sss.split(X, y):
                    
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    #pdb.set_trace()
                    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                    clf.fit(X_train, y_train)   
                    
                    roi_acc.append(clf.score(X_test, y_test))
                    #print(clf.score(X_test, y_test))

                #append each ROI score to 
                roi_decode.append(np.mean(roi_acc))
                print(ss, f'{lr}{rr}', np.mean(roi_acc))
            else:# if roi doesn't exist, make it NAN
                roi_decode.append(np.NaN)

    #pdb.set_trace()
    #append final data to summary
    summary_df = summary_df.append(pd.Series([ss] + roi_decode, index = summary_df.columns), ignore_index= True)
    summary_df.to_csv(f'decoding_summary.csv', index = False)




                #ventral_data[ecn, :] 
        
            #for rr in d_roi:





