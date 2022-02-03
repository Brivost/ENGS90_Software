# To run this script, you need to install: 
#       `pip install tk`
#       `pip install zmq msgpack==0.5.6`
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
import zmq
from analysis import class_to_color
from scipy.spatial import distance
from graphics import *
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from cvd_pupillometry.pyplr.pupil import PupilCore
from cvd_pupillometry.pyplr.utils import unpack_data_pandas
from remote_run import classify, record_data

def calibrate(outdir):
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    grid_w = screen_width/2
    grid_h = screen_height/2

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


# Calibrate at the centroid of each grid
    while not finished:
        for i in range(0,2):
            for j in range(0,2):
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
        
        for point in calibration:
            plt.plot(point[0], point[1], '.')
        
        centroids = [(np.average(grid[0]), np.average(grid[1])) for grid in calibration]
        
        for (x,y) in centroids:
            plt.plot(x,y,'*')
        plt.show()

        head = Text(Point(screen_width/2,screen_height/3), "Finished!").draw(win)
        head.setSize(30)
        head.setStyle('bold')
        sub = Text(Point(screen_width/2,screen_height/3+75), "Press 'r' to recalibrate or any other key to exit").draw(win)
        sub.setSize(20)
        


        if win.getKey() == 'r':
            finished = False
            head.undraw()
            sub.undraw()
            plt.clf()
            centroids = []
            calibration = []
        else:
            finished = True

    win.close()

    with open(outdir + "centroids_2x2.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for (x,y) in centroids:
            writer.writerow([x,y])

    return centroids  


def validate(centroids):
    """
    Estimates accuracy of classifer post-calibration
    
    centroids: calibrated list of centroid positions
    """

    #Set constants    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy() # gets rid of default tk window but also gives strange warning
    
    grid_w = screen_width/2
    grid_h = screen_height/2

    radius = 30         #dot properties
    color = 'black'
    conf_thresh = .7   #confidence threshold
    sampling_rate = 120 # Core collects 120 samples/sec
    capture_time = 1/sampling_rate # Amount of time for Core to collect

    # Set up graphics window
    win = GraphWin("Validation", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height, 0, 0)) # change window position
    
    #Connect to Pupil Core
    p = PupilCore()

    # Set up circle (TOP RIGHT)
    ball = Circle(Point(grid_w/2*3, grid_h/2), radius)
    ball_x_coord = grid_w/2*3 # will be incremented to keep track of ball center
    ball_y_coord = grid_h/2
    ball.setFill(color)
    ball.setOutline(color)
    ball.draw(win)
    
    head = Text(Point(screen_width/2,screen_height/3), "Validation Part 1").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Follow the dot as it traces the letter 'G' " + '\n' + "Press any key to begin").draw(win)
    sub.setSize(20)

    win.getKey()
    head.undraw()
    sub.undraw()

    dx = 3
    dy = 3
    labeled_ball_positions = []
    validation = []
    averaged_validation = []
    classified_grid_number = []

    # top right to top left
    for i in range(round(grid_w/dx)): 
        
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
        
        ball.move(-dx, 0)

        ball_x_coord -= dx
        if ball_x_coord <= grid_w:
            labeled_ball_positions.append(0)
        else:
            labeled_ball_positions.append(1)        

    # top left to bottom left
    for i in range(round(grid_h/dy)): 
        
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
        
        ball.move(0, dy)

        ball_y_coord += dy
        if ball_y_coord <= grid_h:
            labeled_ball_positions.append(0)
        else:
            labeled_ball_positions.append(2)

    # bottom left to bottom right
    for i in range(round(grid_w/dx)): 
       
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
        
        ball.move(dx, 0)

        ball_x_coord += dx
        if ball_x_coord <= grid_w:
            labeled_ball_positions.append(2)
        else:
            labeled_ball_positions.append(3) 

    # bottom right to top right
    for i in range(round(grid_h/dy)): 
        
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
        
        ball.move(0, -dy)

        ball_y_coord -= dy
        if ball_y_coord <= grid_h:
            labeled_ball_positions.append(1)
        else:
            labeled_ball_positions.append(3)
    
    ball.undraw()

    # Average samples if Core took more than one sample at a single ball position
    for sample in validation: 
        sample[0] = np.mean(sample[0]) # Average x
        sample[1] = np.mean(sample[1]) # Average y
        sample[2] = np.mean(sample[2]) # Average conf
        averaged_validation.append(sample)
    
    # Compare ball_positions and validation arrays
    classified_validation = classify(centroids, averaged_validation, conf_thresh)        

    # Extract grid number from classified data
    for sample in classified_validation: 
        classified_grid_number.append(sample[0])

    # Plot correctly and incorrectly classified samples
    for predicted, actual, sample in zip(classified_grid_number, labeled_ball_positions, averaged_validation):
        print(f"Predicted: {predicted}, Actual: {actual}, Sample Coords: {sample[0]}, {sample[1]}")
        if predicted == actual:
            plt.plot(sample[0], sample[1], '.', color = 'green')
        else: 
            plt.plot(sample[0], sample[1], '.', color = 'red')

    # Plot centroids
    for sample in centroids:
        plt.plot(sample[0], sample[1], '*', color=class_to_color(centroids.index(sample)))

    plt.show()

    percent_correct = round((sum(1 for a,b in zip(labeled_ball_positions, classified_grid_number) if a ==b)/len(labeled_ball_positions)) * 100)

    win.close()
    print("Pupil Core is " + str(percent_correct) + "% accurate currently")


def trace_line(centroids):
    #Set constants    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy() # gets rid of default tk window but also gives strange warning
    
    grid_w = screen_width/2
    grid_h = screen_height/2

    radius = 30         #dot properties
    color = 'black'
    conf_thresh = .85   #confidence threshold
    sampling_rate = 120 # Core collects 120 samples/sec
    capture_time = 1/sampling_rate # Amount of time for Core to collect

    # Set up graphics window
    win = GraphWin("Line Trace", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height, 0, 0)) # change window position

    
    #Connect to Pupil Core
    p = PupilCore()

    # Set up circle (CENTER LEFT)
    ball = Circle(Point(0, screen_height/2), radius)
    ball.setFill(color)
    ball.setOutline(color)
    ball.draw(win)
    
    head = Text(Point(screen_width/2,screen_height/3), "Line Trace").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Follow the dot as it traces a line across the screen " + '\n' + "Press any key to begin").draw(win)
    sub.setSize(20)

    win.getKey()
    head.undraw()
    sub.undraw()

    dx = 3
    dy = 3
    
    validation_w = []
    validation_h = []

    #Horizontal glide
    for i in range(round(screen_width/dx)): 
        
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        #validation_w.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]]) 
        validation_w.append(data)
        ball.move(dx, 0)
    
    ball.undraw()

    #Set up circle (CENTER TOP)
    ball = Circle(Point(screen_width/2.0, 0), radius)
    ball.setFill(color)
    ball.setOutline(color)

    ball.draw(win)
    win.getKey()

    for i in range(round(screen_height/dy)): 
        
        # Start recording for 'capture_time' seconds
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
        data = pgr_future.result()
        #validation_h.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]]) 
        validation_h.append(data)
        ball.move(0, dy)
    

    data = [[],[],[]]
    for p_d in validation_w:
        p_data = [np.array([d[b'norm_pos'][0] for d in p_d]), np.array([d[b'norm_pos'][1] for d in p_d]), np.array([d[b'confidence'] for d in p_d])]
   
        if len(p_data) != 1:
            p_data = [sample[np.array(p_data[2]) >= conf_thresh] for sample in p_data]
            p_data = [np.average(sample) for sample in p_data]
                 
        data[0].append(p_data[0])
        data[1].append(p_data[1])
        data[2].append(p_data[2])
            
    try:
        classified_validation_w = classify(centroids, np.array(data).astype(np.float).T, conf_thresh) 
    except ValueError:
        print("Failed case!")

    data = [[],[],[]]
    for p_d in validation_h:
        p_data = [np.array([d[b'norm_pos'][0] for d in p_d]), np.array([d[b'norm_pos'][1] for d in p_d]), np.array([d[b'confidence'] for d in p_d])]
   
        if len(p_data) != 1:
            p_data = [sample[np.array(p_data[2]) >= conf_thresh] for sample in p_data]
            p_data = [np.average(sample) for sample in p_data]
                 
        data[0].append(p_data[0])
        data[1].append(p_data[1])
        data[2].append(p_data[2])
    

    
    try:
        classified_validation_h = classify(centroids, np.array(data).astype(np.float).T, conf_thresh) 
    except ValueError:
        print("Failed case!")


    plt.subplot(1,2,1)
    classified_validation_w = np.array(classified_validation_w).T
    for (c, x, y) in zip(classified_validation_w[0], classified_validation_w[1], classified_validation_w[2]):
        plt.plot(x, y, '.', color=class_to_color(c)) 
    if centroids != None:
        for (x,y) in centroids:
            plt.plot(x,y,'*', color=class_to_color(centroids.index((x,y)))) 
    plt.subplot(1,2,2)
    classified_validation_h = np.array(classified_validation_h).T
    for (c, x, y) in zip(classified_validation_h[0], classified_validation_h[1], classified_validation_h[2]):
            plt.plot(x, y, '.', color=class_to_color(c)) 
    if centroids != None:
        for (x,y) in centroids:
            plt.plot(x,y,'*', color=class_to_color(centroids.index((x,y))))          

    plt.show()



def grid():
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    grid_w = screen_width/2
    grid_h = screen_height/2

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 3     #time for calibration at each point
    conf_thresh = .7   #confidence threshold
    max_deviations = 2 #maximum deviations for outlier trimming

    win = GraphWin("Grid", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height,0,0))

    n = 0
    for i in range(0,2):
        for j in range(0,2):
            #Move the point
            label = Text(Point(grid_w/2+ grid_w*j,grid_h/2+ grid_h*i), str(n))
            label.setSize(30)
            label.setStyle('bold')
            label.draw(win)

            n+=1

    l1 = Line(Point(grid_w, 0), Point(grid_w, screen_height))
    l1.draw(win)

    l3 = Line(Point(0, grid_h), Point(screen_width, grid_h))
    l3.draw(win)

    win.getKey()
    win.close()


if __name__ == "__main__":
    """
    `python remote_run.py -o [output directory] -c [optional: path to csv of calibrated centroids, runs calibration protocol if omitted] 
    -v [if present, run validation protocol] -p [optional: arduino port, defaults to COM4]`

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", metavar=("Output directory"))
    parser.add_argument("-c", metavar=("Path to csv of calibrated centroids"))
    parser.add_argument("-p", metavar=("Arduino port"))
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-l', action='store_true')
    args = parser.parse_args()

    #Make the output directory if it does not exist
    if not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    if args.p == None:
        port = "COM4"
    else: port = args.p

    #Run calibration or load in calibrated centroids as indicated
    if args.c == None:
        centroids = calibrate(args.o)
    else:
        centroids = []
        with open(args.c, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                centroids.append([float(x) for x in row[0].split(',')])
    
    #Run validation function if specified
    if args.v:
        validate(centroids)
    
    #Trace two lines and examine output, if specified
    if args.l:
        trace_line(centroids)
    #Run the experiment
    record_data(args.o, port, centroids)
    

