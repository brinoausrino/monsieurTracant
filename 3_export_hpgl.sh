# add borders
vpype read $1.svg read "border.svg" write "$1.svg"
#convert to hpgl
vpype read $1.svg write --device hp7576a --page-size 70_90 --absolute "$1.hpgl"
python removePenselectCommands.py $1.hpgl