import pandas as pd
import numpy as np 
import pdb
object_cats = ['boat', 'camera', 'car', 'lamp', 'guitar']
cov_dir = "/lab_data/behrmannlab/vlad/docnet//sub-docnet2003/ses-02/covs"

stim_names = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5',
'fix']

all_cov = np.loadtxt(f'{cov_dir}/catmvpa_docnet2003_Run_4_cut.txt', dtype='str')

for ss in stim_names:
    curr_cov = all_cov[all_cov[:,0] == ss,:]
    del_cov = np.delete(curr_cov,0,1)
    np.savetxt(f'{cov_dir}/catmvpa_docnet2003_run4_{ss}.txt', del_cov, fmt ='%s', delimiter = '\t')
    
pdb.set_trace()

for rr in range(1,4):
    cov_file = f'{cov_dir}/CatIdent_mvpa_docnet2003_Run_{rr}_fix.txt'
    curr_cov = np.loadtxt(cov_file, dtype='str')
    del_cov = np.delete(curr_cov,0,1)
    #print(curr)
    new_file = cov_file.replace('CatIdent_mvpa', 'catmvpa')
    new_file = new_file.replace(f'Run_{rr}', f'run{rr}')
    np.savetxt(new_file, del_cov,fmt ='%s', delimiter ='\t')
    for oc in object_cats:
        for obn in range(1,6):
            cov_file = f'{cov_dir}/CatIdent_mvpa_docnet2003_Run_{rr}_{oc}_{obn}.txt'
            curr_cov = np.loadtxt(cov_file, dtype='str')
            del_cov = np.delete(curr_cov,0,1)
            #print(curr)
            new_file = cov_file.replace('CatIdent_mvpa', 'catmvpa')
            new_file = new_file.replace(f'Run_{rr}', f'run{rr}')
            np.savetxt(new_file, del_cov,fmt ='%s', delimiter ='\t')



