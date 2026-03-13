#!/bin/bash
#SBATCH --account=vito
#SBATCH --partition=batch
#SBATCH --job-name=dafoam-tutorial
#SBATCH --output=log.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=8gb
#SBATCH --gpus=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=praharsh.pairaikar@vito.be

module load Singularity

# to speed up sandbox/cleanup
export SINGULARITY_TMPDIR=/tmp/singularity_tmp_${USER}
mkdir -p $SINGULARITY_TMPDIR

SIF_PATH="/home/vito/pairaikp/scratch/dafoam.sif"

singularity run \
    --bind "${PWD}:${PWD}" \
    --contain \
    "${SIF_PATH}" \
    bash -c "
        cd \"${PWD}\" && \
        export HOME=/home/dafoamuser && \
        . /home/dafoamuser/dafoam/loadDAFoam.sh && \
        python trainModel.py
    "
