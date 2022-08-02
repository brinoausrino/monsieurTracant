# set prefix for arguments < 0
DX=$4
DY=$5
if [[ $4 == -* ]] || [[ $5 == -* ]]; then DX="-- "$4; fi

# scales an image and moves it to the correct position of the paper
vpype read $1.svg scaleto $2 $3 linesimplify -t 1 linemerge -t 20 linesort reloop layout 70cmx90cm translate $DX $DY write $1_scaled.svg

# create mask
#vpype read $1_mask.svg scaleto $2 $3 linesimplify -t 1  linesort reloop layout 70cmx90cm translate $DX $DY write $1_scaled_mask.svg
#vpype read $1_mask.svg scaleto $2 $3 layout 70cmx90cm translate $DX $DY write $1_scaled_mask.svg