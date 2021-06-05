import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import warnings
import matplotlib
import matplotlib.pyplot as plt
import fmri_funcs


'''
sub info
'''
study ='docnet'

subj_list=["docnet2003"]

 #runs to pull ROIs from
rois = ["LO_toolloc", 'PFS_toolloc', 'PPC_spaceloc', 'APC_spaceloc']


exp = 'catmvpa' #experimental tasks
exp_suf = ["_12", '_34', '_13', '_24', '_14', '_23']
exp_cond = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5']
exp_cope=list(range(1,26))#copes for localizer runs; corresponding numerically t

for ss in subj_list:
    sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
    raw_dir = f'{sub_dir}/results/beta/{exp}'

    os.makedirs(raw_dir, exist_ok = True) 
    for lr in ['l','r']: #set left and right    
        for rr in rois:
            fmri_funcs.extract_data(sub_dir, raw_dir, f'{lr}{rr}', exp_cond, exp_cope)



