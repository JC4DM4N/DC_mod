for folder in */ ; do
  echo "$folder"
  if [ "$folder" == "CASAFILES/" ]; then
    continue
  else
    cd $folder
    cp ../CASAFILES/getpeakflux.py .
    echo "peak in $folder..."
    python getpeakflux.py --freq 214
    cd ..
  fi
done


