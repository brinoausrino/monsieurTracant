# set prefix for arguments < 0
DX=$4
DY=$5
if [[ $4 == -* ]] || [[ $5 == -* ]]; then DX="-- "$4; fi

# scales an image and moves it to the correct position of the paper
vpype read single_grid.svg scaleto $2 $3 layout 70cmx90cm translate $DX $DY write out/single_grid_scaled.svg

# merge single Item ito grid
if test -f "out/testGrid.svg"; then
    vpype read out/single_grid_scaled.svg read "out/testGrid.svg" write "out/testGrid.svg"
else
    cp "out/single_grid_scaled.svg" "out/testGrid.svg"
fi