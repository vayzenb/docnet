
import pandas as pd
from nilearn import image, plotting, input_data, glm
#from nilearn.glm import threshold_stats_img
import numpy as np

from nilearn.input_data import NiftiMasker
import nibabel as nib


import os
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import matplotlib.pyplot as plt
import pdb
from scipy.stats import gamma
import warnings

warnings.filterwarnings('ignore')


subs = list(range(1001,1013))

study ='spaceloc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
exp = 'spaceloc'
rois = ['PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_toolloc', 'APC_toolloc', 'PPC_distloc', 'APC_distloc']

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()

tr = 1

#load space covs and make it a contrast
vols = 321
runs = list(range(1,7))
runs = [2,4]

brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=0)

def load_filtered_func(run):
    curr_img = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
    #curr_img = image.clean_img(curr_img,standardize=True, t_r=1)
    
    img4d = image.resample_to_img(curr_img,mni)
    
    roi_masker = input_data.NiftiMasker(roi_mask)
    seed_time_series = roi_masker.fit_transform(img4d)
    
    phys = np.mean(seed_time_series, axis= 1)
    phys = (phys - np.mean(phys)) / np.std(phys)
    phys = phys.reshape((phys.shape[0],1))
    
    return img4d, phys

    
def make_psy_cov(run):
    times = np.arange(0, vols, tr
                     )
    curr_cov = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_SA.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
    #contrasting (neg) cov
    curr_cont = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_FT.txt', sep = '\t', header =None, names =['onset','duration', 'value'])
    curr_cont.iloc[:,2] = curr_cont.iloc[:,2] *-1 #make contrasting cov neg
    
    curr_cov = curr_cov.append(curr_cont) #append to positive
    #add number of vols to the timing cols based on what run you are on
    #e.g., for run 1, add 0, for run 2, add 321
    #curr_cov['onset'] = curr_cov['onset'] + ((rn_n)*vols) 
    #pdb.set_trace()
    cov = curr_cov
    #append to concatenated cov
    
    curr_cov = curr_cov.to_numpy()
    
    #convolve to hrf
    psy, name = glm.first_level.compute_regressor(curr_cov.T, 'spm', times)
    
    '''
    #create offsett
    cov['offset'] = cov['onset'] + cov['duration']
    cov[['onset', 'offset']] = np.round(cov[['onset', 'offset']])

    #sort by onset    
    cov = cov.sort_values(by='onset')

    #Convert cov into a continuous predictor with a value for every time point
    psy = np.zeros((vols*rn,2)) #make zero with the length of all runs puttogether
    psy[:,0] = list(range(1,vols*rn+1))

    for ii in cov_ts[:,0]:
        ii =int(ii)
        try:
            psy[ii,1] = cov['value'][(ii >= np.round(cov['onset'])) & (ii < (np.round(cov['offset'])))].to_list()[0]
        except:
            pass
    ''' 
    return psy
    


#runs = [1]

#ss = 1001
#rr = 'rPPC_spaceloc'
def conduct_ppi():
    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        for lr in ['l','r']:
            for rr in rois:
                roi = f'{lr}{rr}'
                if os.path.exists(f'{roi_dir}/{roi}_peak.nii.gz'):
                    print(ss, roi)

                    roi_mask = image.load_img(f'{roi_dir}/{roi}_peak.nii.gz')

                    all_runs = []
                    for rn in runs:
                        confounds = pd.DataFrame(columns =['psy', 'phys'])
                        #load behavioral data
                        #CONVOLE TO HRF
                        psy = make_psy_cov(rn)

                        #load filtered func data
                        img4d, phys = load_filtered_func(rn)

                        #combine phys (seed TS) and psy (task TS) into a regressor
                        confounds['psy'] = psy[:,0]
                        confounds['phys'] =phys[:,0]

                        #create PPI cov by multiply psy * phys
                        ppi = psy*phys
                        ppi = ppi.reshape((ppi.shape[0],1))

                        # extract data from whole brain mask
                        #regress seed phys and psy leaving only residuals

                        brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds])

                        #Correlate interaction term to TS for vox in the brain
                        seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) /
                                          ppi.shape[0])

                        #transform correlation map back to brain
                        seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                        all_runs.append(seed_to_voxel_correlations_img)

                    mean_fc = image.mean_img(all_runs)
                    nib.save(mean_fc, f'{out_dir}/sub-{study}{ss}_{roi}_fc.nii.gz')

            #Extract timeseries from top voxels of roi

            #create interaction term

