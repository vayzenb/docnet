"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""

import subprocess
from glob import glob
import os
import time
import pdb

job_name = 'MVPD_searchlight'
mem = 36
run_time = "2-00:00:00"

#subj info
subj_list = [2001,2002,2003,2004, 2005, 2007, 2008, 2012, 2013, 2014, 2015, 2016]
subj_list = list(range(1001, 1013))

seed_rois = ['lPPC_spaceloc','lAPC_spaceloc']



study_dir = f'/user_data/vayzenbe/GitHub_Repos/docnet'

#the sbatch setup info

def move_files(cl):
    move_files = f"""
    mkdir -p /scratch/vayzenbe/
    rsync -a /lab_data/behrmannlab/image_sets/ShapeNetCore.v2/{cl} /scratch/vayzenbe/
    rsync -a /lab_data/behrmannlab/image_sets/ShapeNet_images /scratch/vayzenbe/
    """
    return move_files

def setup_sbatch(ss, rr):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={job_name}_{ss}_{rr}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p cpu

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:0
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}_{ss}_{rr}.out

module load anaconda3
conda activate brainiak

"""
    return sbatch_setup

#the sbatch cleanup info
sbatch_cleanup = """
rsync -a /scratch/vayzenbe/ShapeNet_images /lab_data/behrmannlab/image_sets/

rm -rf /scratch/vayzenbe/

"""

job_file = f"{job_name}.sh"

for ss in subj_list:
    sub_dir = f'/lab_data/behrmannlab/vlad/docnet/sub-docnet{ss}/ses-02/'
    roi_dir = f'{sub_dir}/derivatives/rois'
    for rr in seed_rois:
        #print(ss, f'{roi_dir}/{rr}.nii.gz')
        if os.path.exists(f'{roi_dir}/{rr}.nii.gz'):
            print(ss, rr)
            #os.remove(f"{job_name}.sh")
            job_cmd = f'python {study_dir}/fmri/docnet_mvpd.py {ss} {rr}'
            f = open(f"{job_name}.sh", "a")
            f.writelines(setup_sbatch(ss,rr))
            f.writelines(job_cmd)
            f.close()
            
            subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
            os.remove(f"{job_name}.sh")








