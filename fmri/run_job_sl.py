import subprocess
from glob import glob
import os
import time
import pdb

job_name = 'MVPD_searchlight'
mem = 36
run_time = "1-00:00:00"

#subj info
subj_list = [2001,2002,2003,2004, 2005, 2007, 2008, 2012, 2013, 2014, 2015, 2016]

seed_rois = ['PPC_spaceloc',  'PPC_toolloc','PPC_distloc', 'APC_spaceloc',  'APC_toolloc','APC_distloc']

subj_list=[2001]
seed_rois = ['PPC_spaceloc']

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

module load anaconda3
conda activate brainiak

# Job name
#SBATCH --job-name={job_name}__{ss}_{rr}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}_{ss}_{rr}.out

"""
    return sbatch_setup

#the sbatch cleanup info
sbatch_cleanup = """
rsync -a /scratch/vayzenbe/ShapeNet_images /lab_data/behrmannlab/image_sets/

rm -rf /scratch/vayzenbe/

"""

job_file = f"{job_name}.sh"

for ss in subj_list:
    for lr in ['r']:
        for rr in seed_rois:
            job_cmd = f'python docnet_mvpd.py {ss} {lr}{rr}'
            f = open(f"{job_name}.sh", "a")
            f.writelines(setup_sbatch(ss,rr))
            f.writelines(job_cmd)
            f.close()
            
            subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
            os.remove(f"{job_name}.sh")








