#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00          # Adjust time as needed
#SBATCH --mem=16GB               # Adjust memory if needed
#SBATCH --job-name=download_csv  # Updated job name for downloading
#SBATCH --output=download_csv_%j.out  # Updated output file name (%j = job ID)

# Purge any existing modules for a clean environment
module purge

# Run wget to download the file
wget -N -c -np "https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/MIMIC-CXR-JPG.csv"