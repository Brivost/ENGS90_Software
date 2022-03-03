# To run this script, you need to install: 
#       `pip install tk`
#       `pip install matplotlib` 
#       `pip install scipy`
#       `pip install pyserial`
#
# Ensure that pip has installed these packages to the PythonPath the IDE is running
# graphics.py and pyplr must be findable by python

import argparse
import csv
import serial
import math
import time
import os

from analysis import class_to_color

from figure_gen import validate, trace_line
from scipy.spatial import distance
from graphics import *
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from cvd_pupillometry.pyplr.pupil import PupilCore
from cvd_pupillometry.pyplr.utils import unpack_data_pandas

def calibrate(outdir, n, two=True):
    """
    Run calibration protocol. Calibrate 3x3 (mounted), 2x2 (mounted) and 3x3 (unmounted) centroids
    """
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    g = n

    grid_w = screen_width/g
    grid_h = screen_height/g

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 3     #time for calibration at each point
    conf_thresh = .7   #confidence threshold
    max_deviations = 2 #maximum deviations for outlier trimming

    win = GraphWin("Calibration", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height,0,0))

    
    head = Text(Point(screen_width/2,screen_height/3), "Calibration").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Focus on each dot for 10 seconds" + '\n' + "Press any key once your eyes are focused on the dot" + '\n' + "Press any key to begin").draw(win)
    sub.setSize(20)

    win.getKey()
    head.undraw()
    sub.undraw()
    
    
#Connect to Pupil Core
    p = PupilCore()

    calibration = []
    traces = []
    finished = False

    num = 0
# Calibrate at the centroid of each grid
    while not finished:
        for i in range(0,g):
            for j in range(0,g):
                #Move the point
                ball = Circle(Point(grid_w/2+ grid_w*j,grid_h/2+ grid_h*i), radius)
                ball.setFill(color)
                ball.setOutline(color)
                ball.draw(win)
                win.getKey()

                #Calibrate

                pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=calibrate_t)
                data = pgr_future.result()
                calibration.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
                ball.undraw()
        


        #Trim off low confidence values
        trimmed = []
        for grid in calibration:
            to_trim = np.array(grid[2]) >= conf_thresh
            trimmed.append([np.array(grid[0])[to_trim],np.array(grid[1])[to_trim], np.array(grid[2])[to_trim]])
        
        calibration = trimmed

        #remove outliers
        trimmed2 = []
        for grid in calibration:
            mean_x = np.mean(grid[0])
            mean_y = np.mean(grid[1])
            grid.append([distance.euclidean((x,y), (mean_x, mean_y)) for (x,y) in zip(grid[0], grid[1])])

            mean = np.mean(grid[3])
            standard_deviation = np.std(grid[3])
            distance_from_mean = abs(grid[3] - mean)
            
            to_trim = distance_from_mean < max_deviations * standard_deviation
            trimmed2.append(([np.array(grid[0])[to_trim],np.array(grid[1])[to_trim], np.array(grid[2])[to_trim]]))
        
        calibration = trimmed2
        # Plot results of calibration
        for point in calibration:
            plt.plot(point[0], point[1], '.')
        
        centroids = [(np.average(grid[0]), np.average(grid[1])) for grid in calibration]
        
        for (x,y) in centroids:
            plt.plot(x,y,'*')
        plt.show()

        head = Text(Point(screen_width/2,screen_height/3), "Finished!").draw(win)
        head.setSize(30)
        head.setStyle('bold')
        sub = Text(Point(screen_width/2,screen_height/3+75), "Press 'r' to recalibrate, 's' to save centroids and record a new set, or any other key to exit").draw(win)
        sub.setSize(20)
        
    #Allow user to recalibrate centroids if insufficient
        key = win.getKey()
        if key == 'r':
            finished = False
            head.undraw()
            sub.undraw()
            plt.clf()
            centroids = []
            calibration = []
        
        elif key == 's':
            finished = False
            head.undraw()
            sub.undraw()
            plt.clf()
            with open(outdir + "centroids_" + str(num) + ".csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for (x,y) in centroids:
                    writer.writerow([x,y])
            
            if num==0:
                head.undraw()
                sub.undraw()
                validate(centroids, win)
            
            num+=1
            centroids = []
            calibration = []
        
        else:
            if num != "2x2":
                with open(outdir + "centroids_" + str(num) + ".csv", 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for (x,y) in centroids:
                        writer.writerow([x,y])
                centroids = []
                calibration = []
                head.undraw()
                sub.undraw()
                finished = False
                g = 2
                grid_w = screen_width/g
                grid_h = screen_height/g
                num = "2x2"
            else:
                finished = True

    win.close()

    with open(outdir + "centroids_" + str(num) + ".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for (x,y) in centroids:
            writer.writerow([x,y])

    return centroids  




def record_data(outdir, port, centroids):
    """ Main function for running tests and recording data
        Returns data from experiment saved as a csv in outdir
        
        outdir: path to directory where csv will be saved
        port: Arduino port. Check gyroscope.c in Arduino (bottom right corner)
        centroids: array of calibrated grid centroid positions

        Each line in the csv file: eye_pos, eye_x, eye_y, eye_conf, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
    """
    
    #Connect to Arduino. Code must be flashed to Arduino prior to running this function
    ard = serial.Serial(port,9600) 
    time.sleep(2)

    #Connect to Pupil Core
    p = PupilCore()

    #Set Constants
    data = [[], [], []]
    raw = []  
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    finished = False
    recording = False
    run_num = 0
    conf_thresh = .7

    #Set up window

    win = GraphWin("Experiment", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height,0,0))

    head = Text(Point(screen_width/2,screen_height/3), "Testing").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Press 'r' to begin a run and 'r' again to end it" + '\n' + "Press 'c' to cancel a run. Data will not be saved" + '\n' + "Press 'q' to end the experiment").draw(win)
    sub.setSize(20)

    rec = Text(Point(screen_width/2,screen_height/3+200), "RECORDING")
    rec.setSize(30)
    rec.setStyle('bold')

    #Collect Data

    while not finished:
        #start_time = time.time()
        key = win.checkKey()
        #Manage the state
        if not recording and key == 'r':
            recording = True
            rec.draw(win)
        elif recording and key == 'r':
            recording = False
            rec.undraw()

            #Process raw data
            data = [[],[],[],[],[],[],[],[],[]]
            for (p_d, accel) in raw:
                p_data = [np.array([d[b'norm_pos'][0] for d in p_d]), np.array([d[b'norm_pos'][1] for d in p_d]), np.array([d[b'confidence'] for d in p_d])]
   
                if len(p_data) != 1:
                    p_data = [sample[np.array(p_data[2]) >= conf_thresh] for sample in p_data]
                    p_data = [np.average(sample) for sample in p_data]
                 
                data[0].append(p_data[0])
                data[1].append(p_data[1])
                data[2].append(p_data[2])

                data[3].append(accel[0])
                data[4].append(accel[1])
                data[5].append(accel[2])
                data[6].append(accel[3])
                data[7].append(accel[4])
                data[8].append(accel[5])

            with open(outdir + str(run_num) + ".csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for line in np.array(data).T:
                    #Error handling, ocassionally Arduino will send non-parseable data
                    try:
                        l = line.astype(np.float)
                        writer.writerow(line)
                    except ValueError:
                        print("Failed case!")
            raw = []
            run_num = run_num+1
        #Cancel run
        elif key == 'c':
            if recording: rec.undraw()
            recording = False
            raw = [] 
        #Quit
        elif key == 'q':
            finished = True
        

        #Read in data if experiment is running
        if recording:
            
           
            accel = ard.readline().decode().strip().split(",") #Read from Arduino

            if len(accel) == 6: #The first reading is often a fragment, dump it

                pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=1/240) #Read from Pupil Core
             
                raw.append((pgr_future.result(),accel)) #Stitch raw data together
            #print("-------- %s -------" % (time.time() - start_time))

    win.close()
    ard.close()
    





if __name__ == "__main__":
    """
    `python remote_run.py -o [output directory] -c [optional: path to csv of calibrated centroids, runs calibration protocol if omitted] 
    -v [if present, run validation protocol] -p [optional: arduino port, defaults to COM4] 
    -n [optional: NxN grid number defaults to 3 (3x3 grid)]`

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", metavar=("Output directory"))
    parser.add_argument("-c", metavar=("Path to csv of calibrated centroids"))
    parser.add_argument("-p", metavar=("Arduino port"))
    parser.add_argument("-n", metavar=("NxN grid number"))
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-l', action='store_true')
    args = parser.parse_args()

    #Make the output directory if it does not exist
    if not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    if args.p == None:
        port = "COM4"
    else: port = args.p

    #Set nxn grid number
    if args.n == None:
        n = 3
    else:
        n = int(args.n)

    #Run calibration or load in calibrated centroids as indicated
    if args.c == None:
        centroids = calibrate(args.o, n)
    else:
        centroids = []
        with open(args.c, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                centroids.append([float(x) for x in row[0].split(',')])
    
    #Run validation function if specified
    if args.v:
        validate(centroids, n)
    
    #Trace two lines and examine output, if specified
    if args.l:
        trace_line(centroids,n)
    #Run the experiment
    record_data(args.o, port, centroids)
