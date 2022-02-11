# ENGS90: Eye Tracking for Seizure Detection
This repo includes all software for Thayer School ENGS90 Group 592: Eye Tracking which allows the user to:
1. Calibrate and collect eye and head tracking data from the Pupil Core eye-tracker and Arduino compatible IMU (<i><b> remote_run.py </b></i>) 
2. Process the hardware-specific output into a hardware-agnostic form, and run analysis using hardware specific features  (<i><b> process_raw.py </b></i>)
3. Use a hardware-agnostic .csv file to analyze collected data (<i><b> analysis.py </b></i>)
<br><br>
Python 3+ required. Neccessary packages for each script noted in the code documentation

## remote_run.py
Calibrate Pupil Core and collect synchronized data from Pupil Core and Arduino IMU. <b> Useage: </b> <br><br>
`python remote_run.py -o [output directory] -c [optional: path to csv of calibrated centroids, runs calibration protocol if omitted] 
    -v [optional: run validation protocol, defaults to False] -p [optional: specifies Arduino port, defaults to COM4] 
    -n [optional: NxN grid number, defaults to 3 (3x3 grid)]`
<br><br>
Detailed experimental protocol for how our data was collected is contained in the Appendix of our final report

## process_raw.py
Serves two functions: 

## analysis.py

## Other contained software and data:

### experiment
Directory containing all the raw recorded data for each of our subjects, designated by behavior. 

### the file where we're gonna dump the figure code that was in remote_run.py
