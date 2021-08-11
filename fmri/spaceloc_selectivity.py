import fmri_funcs
import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import itertools
import matplotlib.pyplot as plt
import warnings
import pdb
import matplotlib
matplotlib.use('Agg')

study ='spaceloc'

subj_list=["spaceloc1001"]

loc_suf = "_spaceloc" #runs to pull ROIs from
exp = ['spaceloc','depthloc','distloc','toolloc'] #experimental tasks
exp_cope=[[4,5], [4,5], [4,5], [6,7]] #copes to test in each ROI; each minus fix
#exp_cope=[[1,2], [1,2], [1,2], [1,2]] #copes to test in each roi; each minus their contrast (e.g., space -feature)

first_runs=[[2,4],[1,2],[1,2],[1,2]] #which first level runs to extract acts for
 #which first level runs to extract acts for

bin_size=100
peak_vox=200
max_vox =2000

cond = [['space','feature'], ['3D',"2D"], ['distance', 'luminance'], ['tool','non_tool']]
#cond = [['space_loc','feature_loc'], ['3D_loc',"2D_loc"], ['distance_loc', 'luminance_loc'], ['tool_loc','non_tool_loc']]

rois = ["LO_toolloc", 'PFS_toolloc', 'PPC_spaceloc', 'APC_spaceloc']


bool_extract_data = True
bool_calc_act = True
bool_calc_mvpa = False


study_dir = f"/lab_data/behrmannlab/vlad/{study}"

def plot_vsf(sub_dir, df, roi,cond, y_ax, save):
    df = df[cond]
    df.columns = cond
    ax = df.plot.line()
    ax.set_xlabel("Number of Voxels")
    ax.set_ylabel(y_ax)
    
    plt.title(roi)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.ioff()
    if save == True:
        plt.savefig(f'{sub_dir}/results/figures/{roi}_VSF.png',bbox_inches='tight')
        plt.close()


def plot_bar(sub_dir, df, roi,cond,save):
    df = df[cond]
    ax = df.plot.bar()
    ax.set_ylabel(f"Mean Beta {peak_vox} voxels")
    
    plt.title(roi)
    plt.tight_layout()
    plt.ioff()
    
    if save == True:
        plt.savefig(f'{sub_dir}/results/figures/{roi}_mean.png',bbox_inches='tight')
        plt.close()

def extract_acts():
    """
    Extracts activation within each ROI from the HighLevel then first level
    """
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
        raw_dir = f'{sub_dir}/results/beta'
        
            
        os.makedirs(raw_dir, exist_ok = True) 
    
        for rr in rois:
            for lr in ['l','r']: #set left and right
                
                roi = f'{lr}{rr}' #set roi
                n = 0
                for ecn, exp_cond in enumerate(exp): #loop through each experiment within each ROI localizer
                    '''
                    Extract acts from HighLevel
                    '''    
                    fmri_funcs.extract_data(sub_dir, raw_dir, roi, exp_cond, cond[ecn],exp_cope[ecn])

                    for run in first_runs[ecn]:
                        '''
                        Extract acts from FirstLevel
                        '''
                        for cn, cope_num in enumerate(exp_cope[ecn]):
                            cope_nifti = f'{sub_dir}/fsl/{exp_cond}/run-0{run}/1stLevel.feat/stats/zstat1_reg.nii.gz'
                            out = f'{raw_dir}/{roi}_{cond[ecn][cn]}_{run}'
                            roi_nifti = f'{sub_dir}/rois/{roi}.nii.gz'
                            bash_cmd  = f'fslmeants -i {cope_nifti} -m {roi_nifti} -o {out}.txt --showall --transpose'
                            print(bash_cmd)
                            #pdb.set_trace()
                            subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
def calc_selectivity():
    '''
    Analyze univariate data
    '''
    roi_cond = ['spaceloc','toolloc']

    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
        raw_dir = f'{sub_dir}/results/beta'
        results_dir = f'{sub_dir}/results/beta_summary'
        os.makedirs(results_dir, exist_ok = True)
        os.makedirs(f'{sub_dir}/results/figures', exist_ok = True)
    
        for rr in rois:
            for lr in ['l','r']: #set left and right

                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    n = 0
                    for ecn, exp_cond in enumerate(exp): #loop through each experiment within each ROI localizer
                        '''
                        Analyze mean for each condition
                        '''
                        for cc in cond[ecn]: #loop through each condition of that localizer
                            curr_df = fmri_funcs.organize_data(sub_dir,raw_dir, roi, cc, 'dist')
                            if n == 0:
                                df = curr_df
                            else:
                                df[cc] = curr_df[cc]
                            n = n+1
                        
                    df.to_csv(f'{results_dir}/{ss}_{roi}_voxel_acts.csv', index = False)
                    
                    cond_name = list(itertools.chain(*cond))
                    df_sum = df.head(peak_vox)
                    df_sum = df_sum.mean()

                    
                    plot_bar(sub_dir, df_sum, roi,cond_name, True)

                    df_roll = df.rolling(bin_size, win_type='triang').mean()
                    df_roll = df_roll.dropna()
                    df_roll= df_roll.reset_index(drop=True)
                    df_roll = df_roll.head(max_vox)

                    plot_vsf(sub_dir,df_roll,roi,cond_name, 'Beta',True)

def calc_mvpa():
    '''
    Analyze MVPA data
    '''
    

    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
        raw_dir = f'{sub_dir}/results/beta'
        results_dir = f'{sub_dir}/results/beta_summary'
        os.makedirs(results_dir, exist_ok = True)
        os.makedirs(f'{sub_dir}/results/figures', exist_ok = True)
    
        for rr in rois:
            for lr in ['l','r']: #set left and right

                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    n = 0
                    for ecn, exp_cond in enumerate(exp): #loop through each experiment within each ROI localizer
                        '''
                        Analyze mean for each condition
                        '''
                        for cc in cond[ecn]: #loop through each condition of that localizer
                            curr_df1 = fmri_funcs.organize_data(sub_dir,raw_dir, roi, f'{cc}_{first_runs[ecn][0]}', 'dist')
                            curr_df2 = fmri_funcs.organize_data(sub_dir,raw_dir, roi, f'{cc}_{first_runs[ecn][1]}', 'dist')
                            #curr_df1.rename(columns= {5: f'{cc}_1'})
                            #curr_df2.rename(columns= {5 : f'{cc}_2'})
                            pdb.set_trace()
                            
                            if n == 0:
                                #df = curr_df1
                                df = curr_df1

                                df[f'{cc}_2'] = curr_df1
                            else:
                                df[f'{cc}_1'] = curr_df1
                                df[f'{cc}_2'] = curr_df1
                            pdb.set_trace()
                            n = n+1
                        
                    df.to_csv(f'{results_dir}/{ss}_{roi}_voxel_acts.csv', index = False)
                    
                    cond_name = list(itertools.chain(*cond))
                    df_sum = df.head(peak_vox)
                    df_sum = df_sum.mean()

                    
                    plot_bar(sub_dir, df_sum, roi,cond_name, True)

                    df_roll = df.rolling(bin_size, win_type='triang').mean()
                    df_roll = df_roll.dropna()
                    df_roll= df_roll.reset_index(drop=True)
                    df_roll = df_roll.head(max_vox)

                    plot_vsf(sub_dir,df_roll,roi,cond_name, 'Beta',True)



#extract_acts()
#calc_selectivity()
calc_mvpa()





