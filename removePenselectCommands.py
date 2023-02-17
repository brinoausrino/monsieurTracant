import argparse
import os
from mmap import mmap
import re

# parse filename
parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('--out', default="", type=str,
                    help='output file')
args = parser.parse_args()

out = args.file
if args.out != "":
    out = args.out
# Opening our text file in read only
# mode using the open() function
with open(args.file, 'r') as file:
  
    # Reading the content of the file
    data = file.read()
    
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
with open(out, 'w') as file:
  
    # Writing the replaced data
    file.write(data)
  
# Printing Text replaced
print("removed pen commands")