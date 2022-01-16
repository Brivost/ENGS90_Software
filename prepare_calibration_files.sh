#!/bin/bash

# This script copies and renames the pupil_positions.csv file for each of 
# the 9 calibration points. It draws from the default export directory 
# of the Pupil Player Software and outputs to a specified output directory. 

# Assuming this script is run same day as calibration data collection
Todays_Date=$(date '+%Y_%m_%d')

# Directory containing calibration recordings (000, 001, etc.)
Calibration_Files_Dir='/Users/Brian/recordings/'${Todays_Date}

# Directory where output csv's should go (up left.csv, up right.csv, etc.)
Output_Dir="/Users/Brian/Documents/ENGS89.90/90/Recordings/Calibration_V2"

Input_Files=(000 001 002 003 004 005 006 007 008)
Output_Files=("up left.csv" "up right.csv" "down left.csv" 
            "down right.csv" "center.csv" "center up.csv" 
            "center down.csv" "center left.csv" "center right.csv")

# Copy and rename files into output directory
for i in ${!Input_Files[*]}; do
    cp ${Calibration_Files_Dir}/${Input_Files[$i]}/exports/000/pupil_positions.csv ${Output_Dir}/"${Output_Files[$i]}"
done