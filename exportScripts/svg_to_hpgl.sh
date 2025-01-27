# add borders
vpype read $1 read $2 write "$1"
#convert to hpgl
vpype read $1 write --device $4 --page-size $5 --absolute "$3"