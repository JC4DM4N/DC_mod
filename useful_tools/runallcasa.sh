#!/bin/bash

for folder in */ ; do
  echo "$folder"
  if [[ "$folder" == "CASAFILES/" ]]; then
    echo "skipping $folder ..."
    continue
  elif [[ "$folder" == "5E-8_"* ]]; then
    cd $folder
    cp ../CASAFILES/fits_paper* .
    echo "doing $folder..."
    python fits_paper.py --freq 127
    python fits_paper_unsharp.py --freq 127
    ###bash casapipeline.sh
    echo "...done $folder"
    cd ..
  fi
done
