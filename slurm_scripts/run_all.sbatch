#!/bin/bash

#SBATCH --job-name rwp_cat
#SBATCH -p test
#SBATCH -o processing_output/rwp_cat_%J.out 
#SBATCH -e processing_output/rwp_cat_%J.err 
#SBATCH --time 04:00:00 
#SBATCH --mem 32000

cd ../scripts/
python shear_profile_filter.py
python shear_profile_normalize.py
