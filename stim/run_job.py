import subprocess
from glob import glob
import os
import time

job_name = 'render_obj'
mem = 24
run_time = "1-00:00:00"
num_obj = 300

model_dir = "/scratch/vayzenbe/ShapeNetCore.v2"
out_dir= '/scratch/vayzenbe/ShapeNet_images'
model_dir = '/lab_data/behrmannlab/image_sets/ShapeNetCore.v2'
out_dir= '/lab_data/behrmannlab/image_sets/ShapeNet_images'
#the sbatch setup info

def move_files(cl):
    move_files = f"""
    mkdir -p /scratch/vayzenbe/
    rsync -a /lab_data/behrmannlab/image_sets/ShapeNetCore.v2/{cl} /scratch/vayzenbe/
    rsync -a /lab_data/behrmannlab/image_sets/ShapeNet_images /scratch/vayzenbe/
    """
    return move_files

def setup_sbatch(cl):
    sbatch_setup = f"""#!/bin/bash -l

    # Job name
    #SBATCH --job-name={job_name}_{cl}

    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=vayzenb@cmu.edu
    
    # Submit job to cpu queue                
    #SBATCH -p gpu

    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
    # Job memory request
    #SBATCH --mem={24}gb

    # Time limit days-hrs:min:sec
    #SBATCH --time {run_time}

    # Standard output and error log
    #SBATCH --output=slurm_out/{job_name}_{cl}.out

    """
    return sbatch_setup

#the sbatch cleanup info
sbatch_cleanup = """
rsync -a /scratch/vayzenbe/ShapeNet_images /lab_data/behrmannlab/image_sets/

rm -rf /scratch/vayzenbe/

"""


cat_folders = glob(f'{model_dir}/*')

job_file = f"{job_name}.sh"


for cln, cl in enumerate(cat_folders):
    exemplar_list = glob(f'{cl}/*')
    
    if len(exemplar_list) > 300:
        job_cmd = f'/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py {cl} {num_obj}'
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(cl))
        f.writelines(job_cmd)
        f.close()

        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")

    #break







