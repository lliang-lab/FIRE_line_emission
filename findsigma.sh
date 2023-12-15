#!/bin/bash -l
#SBATCH --job-name="find_sigma_832"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lliang.uzh@gmail.com
#SBATCH --account=rrg-murray-ac
#SBATCH --time=11:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
N=${SLURM_ARRAY_TASK_ID}
echo ${N}

isnap=832
ihalo=${N}

snapdir=/home/m/murray/lichenli/lichenli/FIRE/FIREBox/snapshot_{isnap}/00${ihalo}/

/gpfs/fs1/home/m/murray/lichenli/venv/bin/python3 findsigma.py -N 1000000 -C ${N} -i ${ihalo} -s 176
sort -k1 -n $snapdir/FB15N1024_00${ihalo}_snap176_sigma_000.txt > $snapdir/FB15N1024_00${ihalo}_snap176_sigma.txt
