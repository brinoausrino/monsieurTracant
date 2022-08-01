# mask image
if test -f "mask.svg"; then
    #vpype read $1_scaled.svg read "mask.svg" occult -i  write "$1_print.svg"
    # create temp folder
    if [ ! -d "temp" ]; then mkdir temp; fi
    # move files to temp folder
    
    cp mask.svg temp
    #mv temp/mask.svg temp/0_mask.svg
    cp $1_scaled.svg temp

    vpype read --layer 1 "temp/$1_scaled.svg" read --layer 2 "temp/mask.svg" write "temp/$1_tmask2.svg"
    #vpype forfile "temp/*.svg"  read --layer %_i+1% %_path% end write "temp/$1_tmask.svg"
    #vpype read "temp/$1_tmask.svg" occult -i write "temp/$1_tmask2.svg"

else
    cp "$1_scaled.svg" "$1_print.svg"
fi

# update mask
if test -f "mask.svg"; then
    vpype read $1_scaled_mask.svg read "mask.svg" linemerge write "mask.svg"
else
    cp "$1_scaled_mask.svg" "mask.svg"
fi