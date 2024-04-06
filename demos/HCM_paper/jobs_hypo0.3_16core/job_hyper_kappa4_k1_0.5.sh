#!/bin/bash
#SBATCH --time=10-00:00:00             # Time limit for the job (REQUIRED).
#SBATCH --job-name=16Ho4_K0.5
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c).
#SBATCH --mem=180G                  # memory required per node - amount of memory (in bytes)
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAL48M192_L# Partition/queue to run the job in. (REQUIRED)
#SBATCH --output=/mnt/gpfs2_4m/scratch/mme250/HCM_paper/hypo0.3_16core/logs/output.%J.out # STDOUT
cd ../../../python_codes
singularity exec --cleanenv /home/mme250/fenics.img  mpiexec -np $SLURM_NTASKS  python MyoFE.py LV_sim /home/mme250/MyoFE/demos/HCM_paper/simulations_hypo0.3_16core/hyper_kappa4_k1_0.5/sim_inputs/input_parameters.json
scontrol show job $SLURM_JOB_ID