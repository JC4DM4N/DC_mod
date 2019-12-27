#!/bin/bash

# The number of nodes and the number of processes per node
# OpenMP cannot request more than one node, and no more than the number of
# cores on the largest node, currently 64, but less for most nodes
#SBATCH -N 1
#SBATCH -n 30

#SBATCH --mem=4G

#SBATCH --job-name=casa

#SBATCH --array=1-30
#SBATCH --output=casalog%a.out

###SBATCH -o casa.log

#SBATCH --ntasks=1

# The wall time for the calculation in HH:MM:SS
#SBATCH -t 12:00:00

# The work directory
cd /disk01/cadman/phantom_runs/torus_runs/SA_CASA

folderid=$SLURM_ARRAY_TASK_ID
i=1

for folder in */ ; do
  if [[ "$folder" == "CASAFILES/" ]]; then
    echo "skipping $folder ..."
    continue
  elif [  $folderid = $i ]; then
    cd $folder
    cp ../CASAFILES/* .
    echo "doing $folder..."
    bash casapipeline.sh
    echo "...done $folder"
    cd ..
    exit
  else
    i=$(($i + 1))
  fi
done

