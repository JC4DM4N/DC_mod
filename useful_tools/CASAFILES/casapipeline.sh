rm *CASA_680GHz_ant8_noisyimage*
/disk01/cadman/programs/casa/bin/casa --nogui -c runcasa.py
python convert_to_mJy.py --freq 680
python fits_paper.py --freq 680
python fits_paper_unsharp.py --freq 680
