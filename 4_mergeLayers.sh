if test -f "merge.svg"; then
    vpype read $1.svg read "merge.svg" write "merge.svg"
else
    cp "$1.svg" "merge.svg"
fi