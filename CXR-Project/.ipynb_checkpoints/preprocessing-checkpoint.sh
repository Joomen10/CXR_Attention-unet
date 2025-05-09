#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00           # adjust as needed
#SBATCH --mem=16GB               # adjust as needed
#SBATCH --job-name=plot_landmark
#SBATCH --output=plot_landmark_%j.out

module purge

singularity exec --nv \
    --overlay /scratch/cjp9314/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python preprocessing.py"