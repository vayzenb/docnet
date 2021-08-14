import subprocess
import glob

job_name = 'render_obj'
mem = 24
time = "1-00:00:00"

model_dir = "/scratch/vayzenbe/ShapeNetCore.v2"
out_dir= '/scratch/vayzenbe/ShapeNet_images'
#the sbatch setup info
sbatch_setup = f""""
\#!/bin/bash -l

\# Job name
\#SBATCH --job-name={job_name}

\#SBATCH --mail-type=ALL
\#SBATCH --mail-user=vayzenb@cmu.edu
 
\# Submit job to cpu queue                
\#SBATCH -p gpu

\#SBATCH --cpus-per-task=6
\#SBATCH --gres=gpu:1
\#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
\# Job memory request
\#SBATCH --mem={24}gb

\# Time limit days-hrs:min:sec
\#SBATCH --time {time}

\# Standard output and error log
\#SBATCH --output={job_name}.out

mkdir -p /scratch/vayzenbe/
rsync -a /lab_data/behrmannlab/image_sets/ShapeNetCore.v2 /scratch/vayzenbe/
rsync -a /lab_data/behrmannlab/image_sets/ShapeNet_images /scratch/vayzenbe/

"""
#the actual job to run
main_job = """
/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py
"""

#the sbatch cleanup info
sbatch_cleanup = """
rsync -a /scratch/vayzenbe/ShapeNet_images /lab_data/behrmannlab/image_sets/

rm -rf /scratch/vayzenbe/

"""


cat_folders = glob(f'{model_dir}/*')

t1 = time.perf_counter()
for cln, cl in enumerate(cat_folders):
    exemplar_list = glob(f'{cl}/*')

    if len(exemplar_list) > 300:
        job_cmd = f'/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py {cl}'


job_file = "test.sh"
f = open( "test.sh", "a")
f.writelines(sbatch_setup)
f.close()
