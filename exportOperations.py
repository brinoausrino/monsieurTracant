import os
from pathlib import Path
from mmap import mmap
import re
import subprocess

paper_formats = {
    "70_90":{
        "size_mm" : [700,900],
        "hpgl_coords":[[-16640, 16640],[-24718, 24718]]
    },
    "A3":{
        "size_mm" : [297,420],
        "hpgl_coords":[[-7060, 7060],[-9984, 9984]]
    }
}

def map(value,  inputMin,  inputMax,  outputMin,  outputMax, clamp = True):

    outVal = ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
    
    if clamp:
        if outputMax < outputMin:
            if outVal < outputMax:
                outVal = outputMax
            elif outVal > outputMin:
                outVal = outputMin
            else:
                if outVal > outputMax:
                    outVal = outputMax
                elif outVal < outputMin:
                    outVal = outputMin

    return outVal
    

def cm_to_hpgl_coord(pos_cm,paperformat):
    return (map(pos_cm[0]*10,paperformat["size_mm"][0]*-0.5,paperformat["size_mm"][0]*0.5,paperformat["hpgl_coords"][0][0],paperformat["hpgl_coords"][0][1]),\
            map(pos_cm[1]*10,paperformat["size_mm"][1]*-0.5,paperformat["size_mm"][1]*0.5,paperformat["hpgl_coords"][1][0],paperformat["hpgl_coords"][1][1]))

def export_polys_as_svg(polys, file, width, height, color = "black"):
    open(file, 'w')\
    .write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\
    <path stroke="'+ color +'" fill="none" d="'+" ".join(["M"+" ".join([f'{x[0]},{x[1]}' for x in y]) for y in polys])+'"/>\
    <line x1="0" x2="1" y1="0" y2="0" stroke="'+ color +'" stroke-width="1"/>\
    <line x1="' + str(width-1) + '" x2="' +str(width) + '" y1="' + str(height-1) + '" y2="' + str(height-1) + '" stroke="'+ color +'" stroke-width="1"/>\
    </svg>')

def place_and_scale_svg(input_file,output_file,width,height,format="70_90",dx="0",dy="0"):
    global paper_formats
    
    bashCommand = (
        "sh exportScripts/scale_position.sh " + input_file + " "
        + output_file + " "
        + width + " "
        + height + " "
        + str(paper_formats[format]["size_mm"][0]) + "mmx" + str(paper_formats[format]["size_mm"][1]) + "mm "
        + dx + " "
        + dy
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def create_border_svg(file,width_cm,height_cm,border,all_four_borders=False):
    cm_to_px = 37.7888888889
    open(file, 'w+')\
    .write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_cm}cm" height="{height_cm}cm">\
    <line x1="' + str(border*cm_to_px) + '" x2="' +str(border*cm_to_px+1) + '" y1="' + str(border*cm_to_px) + '" y2="' + str(border*cm_to_px) + '" stroke="black" stroke-width="1"/>\
    <line x1="' + str((width_cm-border)*cm_to_px-1) + '" x2="' +str((width_cm-border)*cm_to_px) + '" y1="' + str((height_cm-border)*cm_to_px-1) + '" y2="' + str((height_cm-border)*cm_to_px-1) + '" stroke="black" stroke-width="1"/>\
    </svg>')

def hpgl_remove_pen_select_commands(file):
    data = None
    with open(file, 'r') as f:
        # Reading the content of the file
        data = f.read()

        # Searching and replacing the text
        r = re.findall('SP\\d;', data)
        for exp in r:
            data = data.replace(exp, "")

        # Searching and replacing the text
        r2 = re.findall('PS\\d;', data)
        for exp in r2:
            data = data.replace(exp, "")

        # move to top left
        r3 = re.findall(";IN;", data)
        for exp in r3:
            data = data.replace(exp, ";PU0,-12433;PD0,-12233;IN;")

    # Opening our text file in write only
    # mode to write the replaced content
    with open(file, 'w') as f:

        # Writing the replaced data
        f.write(data)

def hpgl_add_signature(file,position,font_size=0.8):
    data = None
    with open(file, 'r') as f:
        
        signature =  ";PU"+str(position[0])+","+str(position[1])+";DT$;SI" + str(font_size*0.625)+"," + str(font_size) + ";LBWTraçant$;IN;"
        
        # Reading the content of the file
        data = f.read()

        # add signature to the end
        r = re.findall(";IN;", data)
        for exp in r:
            data = data.replace(exp, signature)

    # Opening our text file in write only
    # mode to write the replaced content
    with open(file, 'w') as f:

        # Writing the replaced data
        f.write(data)


def svg_to_hpgl(input_file,output_file, format="70_90",add_signature = False,pos_signature=(0,0)):
    global paper_formats

    Path("temp").mkdir(parents=True, exist_ok=True)
    create_border_svg("temp/border_test.svg", paper_formats[format]["size_mm"][0]/10,paper_formats[format]["size_mm"][1]/10,4)
    bashCommand = (
        "sh exportScripts/svg_to_hpgl.sh " + input_file + " temp/border_test.svg "
        + output_file + " "
        + format
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    hpgl_remove_pen_select_commands(output_file)
    if add_signature:
        hpgl_add_signature(output_file, pos_signature)

def export_polys_as_hpgl(polys,image_size,name,width_print,height_print,format="70_90",x_print=0,y_print=0,add_signature = False,output_folder="toPrint"):
    export_polys_as_svg(polys,output_folder + "/" + name + ".svg", image_size[1], image_size[0])
    place_and_scale_svg(output_folder + "/" +name + ".svg",output_folder + "/" +  name + "_scale.svg", 
                        str(width_print) + "mm", 
                        str(height_print) +"mm",
                        format,
                        str(x_print) + "mm",
                        str(y_print) + "mm")
    
    pos_signature = cm_to_hpgl_coord((int(x_print) + int(width_print) - 6, (int(y_print) + int(height_print) - 3)),paper_formats[format])
    svg_to_hpgl(output_folder + "/" + name + "_scale.svg",output_folder + "/" + name + ".hpgl",format,add_signature,pos_signature)
