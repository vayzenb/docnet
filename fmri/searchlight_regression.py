
from nilearn.datasets import MNI152_FILE_PATH
import pandas as pd
from nilearn.image import load_img, get_data, concat_imgs, mean_img, math_img, load_img, get_data, concat_imgs, threshold_img
from nilearn.glm import threshold_stats_img
import numpy as np

from nilearn.input_data import NiftiMasker
import nibabel as nib

from brainiak.searchlight.searchlight import Searchlight, Ball

import time
import os
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask

alpha = .05

subs=["docnet2001", "docnet2002","docnet2003","docnet2005", "docnet2007","docnet2008", "docnet2012"]
copes = list(range(1,26))
study ='docnet'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/searchlight'
exp = 'catmvpa'
os.makedirs(out_dir,exist_ok=True)

model_df = pd.read_csv('/home/vayzenbe/GitHub_Repos/docnet/modelling/rdms/all_rdms.csv')
model_df = model_df[['cornet_s', 'skel']]
model_df=(model_df-model_df.mean())/model_df.std()
whole_brain_mask = get_data(load_mni152_brain_mask())

def calc_rsa(data, sl_mask, myrad, bcvar):
    # Pull out the data
    data4D = data[0]
    
    bolddata_sl = data4D.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data[0].shape[3]).T
    
    brain_rdm = 1-np.corrcoef(bolddata_sl)
    brain_rdm = np.round_(brain_rdm, decimals=6)
    upper = np.triu(brain_rdm,0)
    brain_vec = upper.flatten()
    
    brain_vec = brain_vec[brain_vec!= 0]
    X = bcvar[['skel', 'cornet_s']]
    y = brain_vec
    model = sm.OLS(y, X).fit()

    return model.params['skel']

for ss in subs:
    
    stats_dir= f'{study_dir}/sub-{ss}/ses-02/derivatives/fsl/{exp}/HighLevel.gfeat/'
    #load in all cope images 
    all_nii = []
    for cp in copes:
        all_nii.append(load_img(f'{stats_dir}/cope{cp}.feat/stats/zstat1.nii.gz'))
        
    
    img4d = concat_imgs(all_nii) #compile into 4D
    bold_vol = get_data(img4d) #convert to numpy
    dimsize = img4d.header.get_zooms()  #get dimenisions
    affine = img4d.affine #get affine transforms

    #make brain mask that matches slice prescritpion
    brain_mask = np.zeros(bold_vol[:,:,:,0].shape) 
    brain_mask[bold_vol[:,:,:,0] != 0] = 1
    brain_mask = brain_mask * whole_brain_mask #multiple brain masks to only get slice perscription in gray matter

    #set search light params
    data = bold_vol #data as 4D volume (in numpy)
    mask = brain_mask #the mask to search within
    sl_rad = 4 #radius of searchlight sphere
    max_blk_edge = 5 #how many blocks to send on each parallelized search
    pool_size = 1 #number of cores to work on each search
    bcvar = model_df #any data you need to send to do the analysis in each sphere
    voxels_proportion=.5
    shape = Ball

    print("Setup searchlight inputs")
    print("Subject: ", f'{exp}{ss}')
    print("Input data shape: " + str(data.shape))
    print("Input mask shape: " + str(mask.shape) + "\n")

    sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge, shape = shape, min_active_voxels_proportion= voxels_proportion) #setup the searchlight
    sl.distribute([data], mask) #send the 4dimg and mask
    sl.broadcast(bcvar) #send the relevant analysis vars

    t1 = time.time()
    print("Begin Searchlight\n")
    sl_result = sl.run_searchlight(calc_rsa, pool_size=pool_size)
    print("End Searchlight\n", time.time()-t1)
    #print(sl_result)

    #Save the results to a .nii file
    output_name = f'{out_dir}/{ss}_skel_sl.nii.gz'
    sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
    sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
    sl_nii = nib.Nifti1Image(sl_result, img4d.affine)  # create the volume image
    hdr = sl_nii.header  # get a handle of the .nii file's header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save the volume


'''
Combine all subs
and correct for multiple comparisons
'''
#Reload SLs from each sub
all_nii = []
for ss in subs:
    all_nii.append(load_img(f'{out_dir}/{ss}_skel_sl.nii.gz'))
    
dimsize = all_nii[0].header.get_zooms()  #get dimenisions
affine = all_nii[0].affine #get affine transforms

#average images together
group_img = mean_img(all_nii)
zstat= math_img("(img-np.mean(img))/np.std(img)", img = group_img)

#find fdr-corrected thresholdd
thresh_val = threshold_stats_img(zstat,alpha=alpha, height_control='fdr', cluster_threshold = 5, two_sided = False)
thresh_img = threshold_img(zstat, thresh_val[1])
thresh_img = get_data(thresh_img)
#zero out anything negative
thresh_img[thresh_img[:,:,:] <= 0] = 0

#resave as nifti
thresh_img = thresh_img.astype('double')  # Convert the output into a precision format that can be used by other applications
thresh_img[np.isnan(thresh_img)] = 0  # Exchange nans with zero to ensure compatibility with other applications
thresh_img = nib.Nifti1Image(thresh_img, affine)  # create the volume image
hdr = thresh_img.header  # get a handle of the .nii file's header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))

nib.save(thresh_img, f'{out_dir}/{study}_group_sl.nii.gz')  # Save the volume