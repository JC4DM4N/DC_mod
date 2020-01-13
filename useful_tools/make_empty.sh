#!/bin/bash

for folder in ls SA_runs/*
do
  if [[ "$folder" == *"E-"* ]]; then
    echo ${folder:8}
    mkdir SA_runs_empty/${folder:8}
    cp $folder/continuum* SA_runs_empty/${folder:8}/.
    #cp $folder/IMAGE* SA_runs_empty/${folder:8}/.
    #cp $folder/FINALIMAGE* SA_runs_empty/${folder:8}/.

  fi
done
