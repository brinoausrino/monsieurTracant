from os import walk
from os import system
import time
import os
import shutil
import subprocess
import json

file_list = []

cwd = os.getcwd()
watchfolder = cwd + "/toPrint"
backupfolder = cwd + "/backup"

f = open("config.json")
settings = json.load(f)

class bcolors:
    BLACK = '\x1b[40m'
    BLUEGREEN = '\x1b[46m'
    BLUEVIOLET = '\x1b[45m'
    CYAN = '\033[106m'
    MAGENTA = '\033[105m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_col(text,color):
    if color == "black":
        print(f"{bcolors().BLACK}" + text+ f"{bcolors.ENDC}")
    elif color == "blueGreen":
        print(f"{bcolors().BLUEGREEN}" + text+ f"{bcolors.ENDC}")
    elif color == "blueViolet":
        print(f"{bcolors().BLUEVIOLET}" + text+ f"{bcolors.ENDC}")
    elif color == "cyan":
        print(f"{bcolors().CYAN}" + text+ f"{bcolors.ENDC}")
    elif color == "magenta":
        print(f"{bcolors().MAGENTA}" + text+ f"{bcolors.ENDC}")
    elif color == "bold":
        print(f"{bcolors().BOLD}" + text+ f"{bcolors.ENDC}")
    else:
        print(text)

def update_filelist():
    global file_list
    file_list = []
    filenames = next(walk(watchfolder), (None, None, []))[2]
    if len(filenames) > 0:
            prefix = filenames[0].split("_")[0]
            for file in filenames:
                if prefix in file:
                    file_list.append(file)
            file_list.sort()

def move_files(prefix):
    filenames = next(walk(watchfolder), (None, None, []))[2]
    if not os.path.exists(backupfolder):
        os.mkdir(backupfolder)
    for file in filenames:
        if prefix in file:
            shutil.move(watchfolder + "/" + file,backupfolder + "/" + file)

def create_dialog(filename):
    parts = filename.split("_")
    ret = "Layer "
    ret += parts[1] + "  Farbe: " +parts[2]
    ret += "(Enter) / Layer überspringen(l) / Portrait überspringen (c)"

# main script

while True:
    update_filelist()

    if len(file_list) == 0:
        time.sleep(1)
        clear = lambda: system('clear')
        clear()
        print("")
        print("kein Portrait vorhanden")
    else:
        filename_parts = file_list[0].split("_")
        filename_parts[2] = filename_parts[2].split(".")[0]
        current_layer = int(filename_parts[1])
        layer_color = filename_parts[2].split(".")[0]
            
        clear = lambda: system('clear')
        clear()
        print("")
                
        if current_layer == 0:
            print("neues Portrait verfügbar : " + filename_parts[0])
            
        print_col("Layer " + layer_color + " plotten (Enter)", layer_color)
        print("Layer überspringen (l)")
        v = input("Portrait überspringen (c)")
            
        if v == "":
            print_file = watchfolder + "/" + filename_parts[0]+"_" + filename_parts[1] + "_" + filename_parts[2] +".hpgl"
            bashCommand = "python hp7475a_send.py " + print_file + " -p " + settings["hardware"]["plotterSerial"]
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            #process = subprocess.run(bashCommand.split(), capture_output=True)
            output, error = process.communicate()
            move_files(filename_parts[0] + "_" + filename_parts[1]) 
        elif v == "l":
            move_files(filename_parts[0] + "_" + filename_parts[1])
        elif v == "c":
            move_files(filename_parts[0]) 

