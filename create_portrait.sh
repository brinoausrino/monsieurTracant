# add inkscape layer names for named splittet layers
echo ""
python addInkscapeLayerNames.py $1 --out edit_$1

# scale svg and save single layers to hpgl
echo "scale to 70x90cm"
vpype read edit_$1 scaleto 70cm 90cm layout 70cmx90cm write edit_$1 

echo "export layers"
mkdir -p ${2:-layer}
vpype read edit_$1 linemerge linesort reloop forlayer write --device hp7576a --page-size 70_90 --absolute "${2:-layer}/output_%_name%.hpgl" end

for file in ${2:-layer}/*
do
  python removePenselectCommands.py $file
done