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

<b> Useage: </b> <br><br>
`python process_raw.py -e [experiment directory] -c [centroid file]`
    
 <br>

Relies on the experimental data packaged in a directory with sub-directories subj1,...,subjn corresponding to each subject in the experiment. Each subdirectory contains .csv files named, 'technology', 'eating', 'conversation', 'seizure_right' and seizure_left', along with a set of centroid files, each calibrated in different ways, with suffixes standarded across each subject subdirectory. The user can designate which of these files to use when classifing eye position with the `-c` argument.

<br>
Additionally, performs analysis requiring specific output from the Pupil Core (x, y, and confidence values) such as extracting additional features for LDA analysis. 

## analysis.py
Loads in the experimental data from the hardware-agnostic files from an experimental directory as specified above. Each behavior .csv once converted to a hardware-agnostic format will contain a "data_" prefix, which analysis searches for while loading data.
<br>

<b> Useage: </b> <br><br>
`python analysis.py -e [experiment directory] -o [output directory]`
    
 <br>

### Feature Extraction
Using loaded .csv files, the data is chunked into ~5 second long "epochs" of 130 samples each. The following features are extracted from each epoch:<br><br>
<b> Eye Tracking </b>
* Average classification value
* Number of unique classifications
* Number of changes between classifications from one sample to the next
* Number of low-confidence classifcations
* Percentage of samples in the top and bottom edges of the grid (for 3x3 grid and up)
* Percentage of samples in left and right edges of the grid (for 3x3 grid and up)
<br>

<b> Head Tracking </b>

* Average magnitude of accelerometer readout
* Average magnitude of gyroscope readout
* Standard deviation of accelerometer readout
* Standard deviation of gyroscope readout
* Maximum magnitude of accelerometer readout
* Maximum magnitude of gyroscope readout
* Minimum magnitude of accelerometer readout
* Minimum magnitude of gyroscope readout
* Range of magnitude of accelerometer readout
* Range of magnitude of gyroscope readout

### Linear Discriminant Analysis
The extracted features are then run through a Linear Discriminant Analysis. 5-fold cross validation averages area under the receiver operating characteristic curve (ROC AUC score) and F1 macro score to evaluate the classifier. The following figures are produced:
<br>
![Non-SeizureROC](https://user-images.githubusercontent.com/30049464/156485764-4081f597-2e48-465e-9d86-63ad6526e1fb.png)

<br>
The Receiver Operating Characteristic 

<br>

![Non-SeizureHisto](https://user-images.githubusercontent.com/30049464/156485866-307fda23-028d-4404-a4c2-c91b2098b9be.png)
<br>
LDA Visualization Frequency Histogram
<br>

![Non-SeizureFeaturesTable](https://user-images.githubusercontent.com/30049464/156485944-cfe271eb-7cbc-419a-a604-95f8e3c57ec3.png)
<br>
Ranked Feature Table
<br>
![Non-SeizureFeatures](https://user-images.githubusercontent.com/30049464/156485977-71a30017-1207-491c-8785-8943b60262ac.png)
<br>
Feature Bar Chart
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
Generates a reference grid labeled with classification values.

<img width="626" alt="Screen Shot 2022-02-16 at 8 58 42 PM" src="https://user-images.githubusercontent.com/30049464/154389967-7435238a-c7e5-4ad9-9022-6f653a447afa.png">

