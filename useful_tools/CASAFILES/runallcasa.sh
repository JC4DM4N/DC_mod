b#!/bin/bash

for folder in */ ; do
  echo "$folder"
  if [[ "$folder" == "CASAFILES/" ]]; then
    echo "skipping $folder ..."
    continue
  else
    cd $folder
    cp ../CASAFILES/fits_paper* .
    echo "doing $folder..."
    python fits_paper.py --freq 680
    python fits_paper_unsharp.py --freq 680
    ###bash casapipeline.sh
    echo "...done $folder"
    cd ..
  fi
done
