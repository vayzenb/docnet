import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import warnings
import matplotlib
import matplotlib.pyplot as plt
import fmri_funcs
import seaborn as sns
import pdb
from scipy import stats



'''
sub info
'''
study ='docnet'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
subj_list=["docnet2002", "docnet2007"]
#subj_list=["docnet2003"]

 #runs to pull ROIs from
rois = ["LO_toolloc", 'PFS_toolloc', 'PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'PPC_distloc', 'APC_depthloc', 'APC_distloc', 'PPC_toolloc', 'APC_toolloc',]


exp = 'catmvpa' #experimental tasks
#exp_suf = ["_12", '_34', '_13', '_24', '_14', '_23']
exp_suf = ["_odd", "_even",""]

exp_cond = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5']
exp_cats = ['boat', 'camera',' car', 'guitar','lamp']
exp_cope=list(range(1,26))#copes for localizer runs; corresponding numerically t
suf="_split"


def extract_acts():
    '''
    extract PEs for each condition
    '''
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                for es in exp_suf:
                    print(es)
                    
                    fmri_funcs.extract_data(sub_dir, raw_dir, f'{lr}{rr}', exp,exp_cond, exp_cope,es,'pe1')


def sort_by_functional():
    '''
    load each condition and sort by functional data
    '''
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    for es in exp_suf:
                        
                        n =0
                        for ec in exp_cond:
                            curr_cond = f'{ec}{es}' #combine condition with the suffix of the current highlevel 
                            #combine fROI data with exp data for each condition and sort it by voxel strength/distance
                            curr_df = fmri_funcs.organize_data(sub_dir,raw_dir, roi, curr_cond, 'dist')

                        
                        
                            if n == 0:
                                df = curr_df
                                df = df.rename(columns={f'{ec}{es}': ec})
                            else:
                                df[ec] = curr_df[f'{ec}{es}']
                            n = n+1
                    
                    
                        df = df.iloc[:,5:] #extract just the act columns
                        df = df.sub(df.mean(axis=1), axis=0) #mean center 
                                            
                        df.to_csv(f'{raw_dir}/{roi}_voxel_acts{es}.csv', index = False)


def calc_within_between():
    '''
    load voxel acts from each highlevel and correlate in pairs
    to make asymmetric RDMs
    '''

    #run_pairs = [["_12",'_13', '_14'],['_34','_24', '_23' ]]
    run_pairs = [['_odd'],['_even']]
    
    
    num_vox = 1000
    
    for ss in subj_list:
        summary_df =pd.DataFrame(columns = ['roi', 'identity', 'category','between'])
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'
        results_dir = f'{sub_dir}/results/beta_summary/{exp}'
        os.makedirs(results_dir, exist_ok = True)
        os.makedirs(f'{results_dir}/figures', exist_ok = True)

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                
                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    

                    all_rdms =[]
                    for rpn, rp in enumerate(run_pairs[0]):
                        #load each datafile
                        df1 = pd.read_csv(f'{raw_dir}/{roi}_voxel_acts{run_pairs[0][rpn]}.csv')
                        df1 = df1.iloc[0:num_vox,:]
                        df2 = pd.read_csv(f'{raw_dir}/{roi}_voxel_acts{run_pairs[1][rpn]}.csv')
                        df2 = df2.iloc[0:num_vox,:]

                        rdm = np.zeros((len(exp_cond),len(exp_cond)))
                        #correlate with other runs
                        #This will fill out the entire matrix
                        for c1n, c1 in enumerate(exp_cond):
                            for c2n, c2 in enumerate(exp_cond):
                                #correlate the condition from d1 with df2
                                rdm[c1n, c2n] = 1-np.corrcoef(df1[c1], df2[c2])[0,1]
                                
                        #append RDMs from each run pair
                        all_rdms.append(rdm)
                    #pdb.set_trace()
                    all_rdms = np.array(all_rdms)

                    #average them together
                    comb_rdm = np.mean(all_rdms, axis =0)
                    np.savetxt(f'{results_dir}/{roi}_RDM{suf}.csv', comb_rdm, delimiter=',',fmt='%1.3f')
                    
                    
                    
                    #save plot
                    sns_plot  = sns.heatmap(comb_rdm, linewidth=0.5)
                    sns_plot.figure.savefig(f'{results_dir}/figures/{roi}_rdm{suf}.png')
                    plt.close()

                    #Pull out within-ident
                    ident_mat = np.identity(len(exp_cond))
                    ident_rdm = ident_mat * comb_rdm
                    ident_rdm[ident_rdm==0] = np.nan
                    ident_mean = np.nanmean(ident_rdm)
                    #ident_se = stats.sem(ident_rdm, nan_policy = 'omit')

                    #pull out between-cat
                    between_rdm = comb_rdm
                    np.fill_diagonal(between_rdm,0)
                    between_rdm[between_rdm==0] = np.nan
                    cat_means =[]
                    
                    #loop through category blocks
                    for ii in range(0,len(exp_cond), len(exp_cats)):
                        #extract corrs for a category
                        curr_cat = between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)]
                        #append the mean for that category
                        cat_means.append(np.nanmean(curr_cat))
                        #replace that category with nans so you can average later
                        between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)] = np.nan

                    cat_mean = np.mean(cat_means)
                    #cat_se = stats.sem(cat_means, nan_policy = 'omit')
                    between_mean =  np.nanmean(between_rdm)
                    #between_se =   stats.sem(between_rdm, nan_policy = 'omit')
                    
                    summary_df = summary_df.append(pd.Series([roi, ident_mean, cat_mean, between_mean],index = summary_df.columns),ignore_index=True)
                    summary_df.to_csv(f'{results_dir}/mvpa_summary{suf}.csv', index = False)
                    print(ss, {lr},rr)


def calc_summary_rdms():
    summary_df =pd.DataFrame(columns = ['roi', 'identity', 'category','between', 'identity_se', 'category_se','between_se'])

    summary_dir = f'{study_dir}/derivatives/results/{exp}'
    os.makedirs(summary_dir, exist_ok = True)
    os.makedirs(f'{summary_dir}/figures', exist_ok = True)

    for lr in ['l','r']: #set left and right    
        for rr in rois:
            
            roi = f'{lr}{rr}' #set roi
        
            all_rdms = []
            for ss in subj_list:
                sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
                results_dir = f'{sub_dir}/results/beta_summary/{exp}'


                all_rdms.append(np.loadtxt(f'{results_dir}/{roi}_RDM{suf}.csv',delimiter=',',dtype='float'))
                

                #pdb.set_trace()
            all_rdms = np.array(all_rdms)

            #average them together
            comb_rdm = np.mean(all_rdms, axis =0)
            np.savetxt(f'{summary_dir}/{lr}{roi}_RDM{suf}.csv', comb_rdm, delimiter=',')
            
            
            #save plot
            sns_plot  = sns.heatmap(comb_rdm, linewidth=0.5)
            sns_plot.figure.savefig(f'{summary_dir}/figures/{roi}_rdm{suf}.png')
            plt.close()

            '''
            figure out error bars!!
            '''

            #Pull out within-ident
            ident_mat = np.identity(len(exp_cond))
            ident_rdm = ident_mat * comb_rdm
            ident_rdm[ident_rdm==0] = np.nan
            
            ident_rdm = ident_rdm.flatten()
            ident_rdm = ident_rdm[~np.isnan(ident_rdm)]
            ident_mean = np.nanmean(ident_rdm)
            ident_se = stats.sem(ident_rdm)
            
            #ident_se = stats.sem(ident_rdm, nan_policy = 'omit')

            #pull out between-cat
            between_rdm = comb_rdm

            #fill diagnol (within) zeros/nans
            np.fill_diagonal(between_rdm,0)
            between_rdm[between_rdm==0] = np.nan
            cat_means =[]
            cat_sems = []
            #loop through category blocks
            for ii in range(0,len(exp_cond), len(exp_cats)):
                #extract corrs for a category by moving in 5 step increments
                curr_cat = between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)]
                curr_cat = curr_cat.flatten()
                curr_cat = curr_cat[~np.isnan(curr_cat)]
                #append the mean for that category
                cat_means.append(np.nanmean(curr_cat))
                cat_sems.append(stats.sem(curr_cat))
                #cat_se.append
                #replace that category with nans so you can average later
                between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)] = np.nan

            
            #between_rdm = between_rdm.flatten()
            #between_rdm = between_rdm[~np.isnan(between_rdm)]

            cat_mean = np.mean(cat_means)
            cat_se = np.mean(cat_sems)

            between_rdm = between_rdm.flatten()
            between_rdm = between_rdm[~np.isnan(between_rdm)]
            #cat_se = stats.sem(cat_means, nan_policy = 'omit')
            between_mean =  np.nanmean(between_rdm)
            between_se = stats.sem(between_rdm)
            #between_se =   stats.sem(between_rdm, nan_policy = 'omit')
            #pdb.set_trace()
            summary_df = summary_df.append(pd.Series([roi, ident_mean, cat_mean, between_mean, ident_se, cat_se, between_se],index = summary_df.columns),ignore_index=True)
    summary_df.to_csv(f'{summary_dir}/mvpa_summary_rdms{suf}.csv', index = False)
    print(ss, {lr},rr)


def calc_summary_mvpa():
    '''
    Combine within/between results
    '''        
    #note: each sub might have different rois

    summary_df =pd.DataFrame(columns = ['roi', 'identity', 'category','between', 'identity_se', 'category_se','between_se'])

    summary_dir = f'{study_dir}/derivatives/results/{exp}'
    os.makedirs(summary_dir, exist_ok = True)
    os.makedirs(f'{summary_dir}/figures', exist_ok = True)
    suf = '_split'
    for lr in ['l','r']: #set left and right    
        for rr in rois:
            roi_vals = []
            for ss in subj_list:
                sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
                results_dir = f'{sub_dir}/results/beta_summary/{exp}'
                roi_nifti = f'{sub_dir}/rois/{lr}{rr}.nii.gz'

                if os.path.exists(roi_nifti):
                    #load df
                    
                    df = pd.read_csv(f'{results_dir}/mvpa_summary{suf}.csv')
                    #extract and append data from current roi
                    curr_roi = df.loc[df['roi'] == f'{lr}{rr}',:].values.flatten().tolist()
                    roi_vals.append(curr_roi[1:])

                    #roi_vals.append(df.loc[df['roi'] == f'{lr}{rr}',:].values.flatten().tolist())
            roi_vals = np.array(roi_vals)

            roi_means = pd.Series([f'{lr}{rr}'] + np.mean(roi_vals, axis = 0).tolist() + stats.sem(roi_vals, axis = 0).tolist(),index = summary_df.columns)
            summary_df = summary_df.append(roi_means,ignore_index=True)
    summary_df.to_csv(f'{summary_dir}/mvpa_summary{suf}.csv', index = False)
    #pdb.set_trace()
    #plt.bar(summary_df['roi'], df_mean,  yerr=df_se)
            #pdb.set_trace()




extract_acts()
sort_by_functional()
calc_within_between()
#calc_summary()
calc_summary_mvpa()






                





                    





