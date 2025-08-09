#!/bin/bash
#SBATCH --job-name=variability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH -o /home/skojima/output_job/erp-dl/job-%x.%N.%j.out
#SBATCH -e /home/skojima/output_job/erp-dl/job-%x.%N.%j.err
#SBATCH --mail-user=simon.kojima@inria.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --nodelist=sirocco21

source /beegfs/skojima/miniconda3/etc/profile.d/conda.sh
conda activate erp-dl
which python
which conda
conda env list

python kojima2024b.py