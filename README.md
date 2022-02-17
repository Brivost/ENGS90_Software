# ENGS90: Eye Tracking for Seizure Detection
This repo includes all software for Thayer School ENGS90 Group 592: Eye Tracking which allows the user to:
1. Calibrate and collect eye and head tracking data from the Pupil Core eye-tracker and Arduino compatible IMU (<i><b> remote_run.py </b></i>) 
2. Process the hardware-specific output into a hardware-agnostic form, and run analysis using hardware specific features  (<i><b> process_raw.py </b></i>)
3. Use a hardware-agnostic .csv file to analyze collected data (<i><b> analysis.py </b></i>)
<br><br>
Python 3+ required. Neccessary packages for each script noted in the code documentation

## remote_run.py
Calibrates Pupil Core and collects synchronized data from Pupil Core and Arduino IMU. Writes out centroid coordinates and raw experimental data to a designated directory. The user is able to control the process via a simple GUI.


<b> Useage: </b> <br><br>
`python remote_run.py -o [output directory] -c [optional: path to csv of calibrated centroids] 
    -v [optional: run validation protocol, defaults to False] -p [optional: specifies Arduino port, defaults to COM4] 
    -n [optional: NxN grid number, defaults to 3 (3x3 grid)]`
    ``
   <br>
![testingscreen](https://user-images.githubusercontent.com/30049464/154166026-ef23e8b5-2370-40fd-bf35-15ccd7e67223.png)

* Pressing 'r' begins and ends an experimental run, saving the data to the directory with the `-o` argument. <br>
* Pressing 'c' will cancel the current run -- no data will be saved. <br>
* Pressing 'q' will end the experiment and the GUI will close <br>
* While data is being recorded, a bold "<b> RECORDING </b> " will show up beneath the text shown above


<br><br>
Detailed experimental protocol for calibrating the Pupil Core and collecting data is contained in the Appendix of the Final Report

## process_raw.py
Processes the raw output files written by remote_run.py into a hardware-agnostic .csv file of the form: <br><br>

Eye position classification, Acceleration_X, Acceleration_Y, Acceleration_Z, Gyroscope_X, Gyroscope_Y, Gyroscope_Z
<br><br>

Performs analysis specific 

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
