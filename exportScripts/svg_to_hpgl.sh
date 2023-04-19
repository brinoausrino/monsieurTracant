# add borders
vpype read $1 read $2 write "$1"
#convert to hpgl
vpype read $1 write --device hp7576a --page-size $4 --absolute "$3"