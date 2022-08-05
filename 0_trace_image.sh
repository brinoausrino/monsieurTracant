# trace
potrace --svg $1.bmp -o $1.svg

# add borders for scale
vpype read $1.svg read "border_single.svg" write "$1.svg"
