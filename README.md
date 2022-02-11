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
Detailed experimental protocol for how data was collected is contained in the Appendix of the Final Rport

## process_raw.py
Serves two functions: 

## analysis.py

## experiment
Directory containing all the raw recorded data for each of our subjects, designated by behavior. 

## figure_gen.py

Contains code for generating the supplemental figures used in the Final Report.

### trace_line
Run via the `-l` argument from `remote_run.` Graphs eye position data colored by classification value with colored centroids for reference 
![traceline2](https://user-images.githubusercontent.com/30049464/153648038-29b0546f-25a9-4e4f-9a48-a729e0f77643.png)

### validate
Run as default during the calibration process to benchmark the accuracy of the calibrated centroids
![validation_with_centroids](https://user-images.githubusercontent.com/30049464/153649514-4be26d07-f0f9-4c25-a032-ae4292796416.png)

### plot_accel
Generates a figure of gyroscope and accelerometer readout. xlim and ylim values may need to be manually adjusted
![nod](https://user-images.githubusercontent.com/30049464/153649782-432fd902-2a2f-4d7d-b438-914338c7b547.png)


### grid
Generates a reference grid labeled with classification values
