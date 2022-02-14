import argparse
import csv
import serial
import math
import time
import os
import zmq

from scipy.spatial import distance
from graphics import *
from process_raw import classify
from analysis import class_to_color
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from cvd_pupillometry.pyplr.pupil import PupilCore
from cvd_pupillometry.pyplr.utils import unpack_data_pandas



def trace_line(centroids, n):
    #Set constants    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy() # gets rid of default tk window but also gives strange warning
    
    grid_w = screen_width/n
    grid_h = screen_height/n

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
    ball = Circle(Point(0, screen_height/2.0), radius)
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

    # Set up circle (CENTER TOP)
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
            plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y]))) 
    plt.subplot(1,2,2)
    classified_validation_h = np.array(classified_validation_h).T
    for (c, x, y) in zip(classified_validation_h[0], classified_validation_h[1], classified_validation_h[2]):
            plt.plot(x, y, '.', color=class_to_color(c)) 
    if centroids != None:
        for (x,y) in centroids:
            plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y])))          

    plt.show()



def grid():
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    grid_w = screen_width/3
    grid_h = screen_height/3

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 3     #time for calibration at each point
    conf_thresh = .7   #confidence threshold
    max_deviations = 2 #maximum deviations for outlier trimming

    win = GraphWin("Grid", screen_width, screen_height)
    win.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height,0,0))

    n = 0
    for i in range(0,3):
        for j in range(0,3):
            #Move the point
            label = Text(Point(grid_w/2+ grid_w*j,grid_h/2+ grid_h*i), str(n))
            label.setSize(30)
            label.setStyle('bold')
            label.draw(win)

            n+=1

    l1 = Line(Point(grid_w, 0), Point(grid_w, screen_height))
    l1.draw(win)
    l2 = Line(Point(grid_w*2, 0), Point(grid_w*2, screen_height))
    l2.draw(win)

    l3 = Line(Point(0, grid_h), Point(screen_width, grid_h))
    l3.draw(win)

    l4 = Line(Point(0, grid_h*2), Point(screen_width, grid_h*2))
    l4.draw(win)
    win.getKey()
    win.close()

def validate(centroids, window, n=3):
    """
    Estimates accuracy of classifer post-calibration
    
    centroids: calibrated list of centroid positions
    """

    print(centroids)

    #Set constants    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy() # gets rid of default tk window but also gives strange warning
    
    grid_w = screen_width/n
    grid_h = screen_height/n

    radius = 30         #dot properties
    color = 'black'
    conf_thresh = .7   #confidence threshold
    sampling_rate = 120 # Core collects 120 samples/sec
    capture_time = 1/sampling_rate # Amount of time for Core to collect

    # Set up graphics window
    # window = GraphWin("Validation", screen_width, screen_height)
    # window.master.geometry('%dx%d+%d+%d' %(screen_width, screen_height, 0, 0)) # change window position
    
    # #Connect to Pupil Core
    p = PupilCore()

        
    finished=False
    while not finished:
    # Set up circle (TOP LEFT)
        ball = Circle(Point(grid_w/2, grid_h/2), radius)
        ball_x_coord = grid_w/2 # will be incremented to keep track of ball center
        ball_y_coord = grid_h/2
        ball.setFill(color)
        ball.setOutline(color)
        ball.draw(window)
        
        head = Text(Point(screen_width/2,screen_height/3), "Validation").draw(window)
        head.setSize(30)
        head.setStyle('bold')
        sub = Text(Point(screen_width/2,screen_height/3+75), "Follow the dot as it traces the letter 'S' " + '\n' + "Press any key to begin").draw(window)
        sub.setSize(20)

        window.getKey()
        head.undraw()
        sub.undraw()

        left_to_right = True
        dx = 3
        dy = 3

        labeled_ball_positions = []
        validation = []
        averaged_validation = []
        classified_grid_number = []

        for i in range(n-1): 

            #side to side
            for j in range(abs(round((screen_width-(grid_w))/dx))):
                # Start recording for 'capture_time' seconds
                pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
                data = pgr_future.result()
                validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])

                ball.move(dx, 0)

                ball_x_coord += dx
                labeled_ball_positions.append(math.floor(ball_x_coord/grid_w) + (n*i))
        
            #down one row
            for j in range(abs(round(grid_h/dy))): 
            
                # Start recording for 'capture_time' seconds
                pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
                data = pgr_future.result()
                validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
            
                ball.move(0, dy)

                ball_y_coord += dy
                labeled_ball_positions.append((n*(math.floor(ball_y_coord/(grid_h*(i+1))))) + (((math.floor(ball_x_coord/grid_w)) + (n*i))))


            
            if (left_to_right):
                left_to_right = False
                dx = -3
            else:
                left_to_right = True
                dx = 3 

        if (n % 2) == 0:
            dx = -3
        else:
            dx = 3

        #side to side one more time
        for j in range(abs(round((screen_width-(grid_w))/dx))):
            # Start recording for 'capture_time' seconds
            pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=capture_time)
            data = pgr_future.result()
            validation.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])

            ball.move(dx, 0)

            ball_x_coord += dx
            labeled_ball_positions.append(math.floor(ball_x_coord/grid_w) + (n*(n-1)))        
        
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
        print("Pupil Core is " + str(percent_correct) + "% accurate currently")
        
        if window.getKey() != 'r':
            finished = True