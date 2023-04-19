# set prefix for arguments < 0
DX=$6
DY=$7
if [[ $6 == -* ]] || [[ $7 == -* ]]; then DX="-- "$6; fi

# scales an image and moves it to the correct position of the paper
vpype read $1 scaleto $3 $4 linesimplify -t 1 linesort reloop layout $5 translate $DX $DY rotate 270 write $2
#vpype read $1 scaleto $3 $4 linesimplify -t 1 linemerge -t 20 linesort reloop layout $5 translate $DX $DY rotate 270 write $2
# create mask
#vpype read $1_mask.svg scaleto $2 $3 linesimplify -t 1  linesort reloop layout 70cmx90cm translate $DX $DY write $1_scaled_mask.svg
#vpype read $1_mask.svg scaleto $2 $3 layout 70cmx90cm translate $DX $DY write $1_scaled_mask.svg