'''
Do conjunction analysis by converting thresholded ROIs into proportions
space / (space + other)

'''

import subprocess
import os

anat = ['PPC', 'APC']
subj_list=["spaceloc1001","spaceloc1002","spaceloc1003","spaceloc1004","spaceloc1005", "spaceloc1006",
"spaceloc1007","spaceloc1008", "spaceloc1009","spaceloc1010", "spaceloc1011","spaceloc1012"]

subj_list=["spaceloc1001"]


exp="spaceloc"
cond=["depthloc", "distloc", "toolloc"]
loc_suf ='_roi'
cope_num = 1

func_roi = []
d_roi = ['LOC','PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_distloc',  'APC_distloc', 'PPC_toolloc', 'APC_toolloc']

exp_dir=f"/lab_data/behrmannlab/vlad/spaceloc"

#bash_cmd  = f'fslmeants -i {cope_nifti} -m {roi_nifti} -o {out}.txt --showall --transpose'

for cc in cond:
    roi_dir = f'{exp_dir}/derivatives/rois'
    os.makedirs(f'{roi_dir}/conjunction', exist_ok=True)
    os.makedirs(f'{roi_dir}/thresh', exist_ok=True)
    
    space_stat = f'{exp_dir}/derivatives/fsl/HighLevel_spaceloc.gfeat/cope1.feat/'
    cond_stat =f'{exp_dir}/derivatives/fsl/HighLevel_{cc}.gfeat/cope1.feat/'

    '''
    #thresh each image by multiplying it by the cluster-corrected mask
    #thresh space image
    bash_cmd = f'fslmaths {space_stat}/cluster_mask_zstat1.nii.gz -bin -mul {space_stat}/stats/zstat1.nii.gz {roi_dir}/thresh/spaceloc_thresh.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    #thresh cond image
    bash_cmd = f'fslmaths {cond_stat}/cluster_mask_zstat1.nii.gz -bin -mul {cond_stat}/stats/zstat1.nii.gz {roi_dir}/thresh/{cc}_thresh.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)
    '''

    '''
    This version creates binary conjuction maps
    '''
    #Make binary cluster-corrected mask
    #add 1 to space mask so it can seperate from the other
    bash_cmd = f'fslmaths {space_stat}/cluster_mask_zstat1.nii.gz -bin {roi_dir}/conjunction/spaceloc_sep.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    #make other condition cluster mask binary
    #leave it as 1 so it's in a diff range
    bash_cmd = f'fslmaths {cond_stat}/cluster_mask_zstat1.nii.gz -bin {roi_dir}/conjunction/{cc}_sep.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    #make conjunction by adding cluster masks
    bash_cmd = f'fslmaths {roi_dir}/conjunction/spaceloc_sep.nii.gz -add {roi_dir}/conjunction/{cc}_sep.nii.gz -thr 1.5 {roi_dir}/conjunction/spaceloc_{cc}_conj_bin.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    '''
    THis version makes continous conjuction maps
    '''
    #make the whole z-map into a proportion and then mul by the cluster images
    
    #make denominmator for final image
    bash_cmd = f'fslmaths {space_stat}/stats/zstat1.nii.gz -add {cond_stat}/stats/zstat1.nii.gz {roi_dir}/conjunction/spaceloc_{cc}_denom.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    #make spaceloc proportion
    bash_cmd = f'fslmaths {space_stat}/stats/zstat1.nii.gz -div {roi_dir}/conjunction/spaceloc_{cc}_denom.nii.gz -mul {roi_dir}/conjunction/spaceloc_sep.nii.gz {roi_dir}/conjunction/spaceloc_{cc}_conj_cont.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)

    #make proportion for other cond
    bash_cmd = f'fslmaths {cond_stat}/stats/zstat1.nii.gz -div {roi_dir}/conjunction/spaceloc_{cc}_denom.nii.gz -mul {roi_dir}/conjunction/{cc}_sep.nii.gz {roi_dir}/conjunction/{cc}_spaceloc_conj_cont.nii.gz'
    subprocess.run(bash_cmd.split(), check=True)



    for ss in subj_list:
        sub_dir = f"{exp_dir}/sub-{ss}/ses-01/derivatives"

        roi_dir = f'{sub_dir}/rois'
        os.makedirs(f'{roi_dir}/conjunction', exist_ok=True)
        os.makedirs(f'{roi_dir}/thresh', exist_ok=True)

        space_stat =  f'{sub_dir}/fsl/spaceloc/HighLevel_roi_2runs.gfeat/cope{cope_num}.feat/'

    
        func_dir = f'{sub_dir}/fsl/{cc}/HighLevel{loc_suf}.gfeat'
        cond_stat = f'{func_dir}/cope{cope_num}.feat/'
        
        #NEED TO THRESHOLD
            #thresh each image by multiplying it by the cluster-corrected mask
        #thresh space image
        bash_cmd = f'fslmaths {space_stat}/cluster_mask_zstat1.nii.gz -bin -mul {space_stat}/stats/zstat1.nii.gz {roi_dir}/thresh/spaceloc_thresh.nii.gz'
        subprocess.run(bash_cmd.split(), check=True)

        #thresh cond image
        bash_cmd = f'fslmaths {cond_stat}/cluster_mask_zstat1.nii.gz -bin -mul {cond_stat}/stats/zstat1.nii.gz {roi_dir}/thresh/{cc}_thresh.nii.gz'
        subprocess.run(bash_cmd.split(), check=True)
        
        #make denominmator for final image
        bash_cmd = f'fslmaths {roi_dir}/thresh/spaceloc_thresh.nii.gz -add {roi_dir}/thresh/{cc}_thresh.nii.gz {roi_dir}/conjunction/spaceloc_{cc}_conj.nii.gz'
        subprocess.run(bash_cmd.split(), check=True)

        #make proportion
        bash_cmd = f'fslmaths {roi_dir}/thresh/spaceloc_thresh.nii.gz -div {roi_dir}/conjunction/spaceloc_{cc}_conj.nii.gz -add 1 {roi_dir}/conjunction/spaceloc_{cc}_conj.nii.gz'
        subprocess.run(bash_cmd.split(), check=True)




