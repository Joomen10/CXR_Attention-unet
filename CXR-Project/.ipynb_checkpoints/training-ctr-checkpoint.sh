#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00          # Increase if training needs more than 1 hour
#SBATCH --mem=16GB               # Increase memory if needed
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=attention_unet_CXR_withCTR  # Job name
#SBATCH --output=attention_unet_CXR_withCTR_%j.out  # Output file name (%j = job ID)

# Purge any existing modules for a clean environment
module purge

# Run your script inside the Singularity container
singularity exec --nv \
    --overlay /scratch/cjp9314/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python training-ctr.py"