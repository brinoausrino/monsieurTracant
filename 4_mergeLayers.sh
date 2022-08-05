if test -f "out/merge.svg"; then
    vpype read $1.svg read "out/merge.svg" write "out/merge.svg"
else
    cp "$1.svg" "out/merge.svg"
fi