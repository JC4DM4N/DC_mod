#!/bin/bash

for folder in *amax*
do
  cd $folder
  if test -f stokes.dat
  then
    rm stokes.dat
  fi

  if test -f casa-*.log
  then
    rm casa-*.log
  fi

  i=0
  #remove initial grid file.
  #then only keep the final 2 lucy grid files.
  if test -f cass_grid.grid
  then
    rm cass_grid.grid
  fi
  for file in $(ls lucy_* | sort -r)
  do
    i=$((i+1))
    if [ $i -gt 2 ]
    then
      echo deleting $folder/$file
      rm $file
    fi
  done
  cd ..
done
