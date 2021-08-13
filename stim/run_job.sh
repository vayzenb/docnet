#!/bin/bash -l

# Job name
#SBATCH --job-name=blender_sample
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL 
##############################################
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
 
# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
# Job memory request
#SBATCH --mem=24gb

# Time limit days-hrs:min:sec
#SBATCH --time 1-00:00:00

# Standard output and error log
#SBATCH --output=blender_out.out

mkdir -p /scratch/vayzenbe/

rsync -a /lab_data/behrmannlab/image_sets/ShapeNetCore.v2 /scratch/vayzenbe/
rsync -a /lab_data/behrmannlab/image_sets/ShapeNet_images /scratch/vayzenbe/

/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py

rsync -a /scratch/vayzenbe/ShapeNet_images /lab_data/behrmannlab/image_sets/

rm -rf /scratch/vayzenbe/
