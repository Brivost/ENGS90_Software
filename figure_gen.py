import argparse
import csv
import serial
import math
import time
import os
import zmq

from scipy.spatial import distance
from graphics import *
from process_raw import classify, load_centroids
from analysis import class_to_color
from itertools import product
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from cvd_pupillometry.pyplr.pupil import PupilCore
from cvd_pupillometry.pyplr.utils import unpack_data_pandas

def class_to_color(c):
    """
    Turn classification value to a color
    """
    if   c==0: return 'red'
    elif c==1: return 'pink'
    elif c==2: return 'orange'
    elif c==3: return 'yellow'
    elif c==4: return 'green'
    elif c==5: return 'blue'
    elif c==6: return 'brown'
    elif c==7: return 'purple'
    elif c==8: return 'black'


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

def plot_centroids(centroids, n=3):
    for (x,y) in centroids:
        plt.plot(x,y,'*', color='green', label="Original")
    
    diag_x = [ [centroids[0][0] , centroids[4][0], centroids[8][0]], [centroids[2][0], centroids[4][0], centroids[6][0]] ]
    diag_y = [ [centroids[0][1] , centroids[4][1], centroids[8][1]], [centroids[2][1], centroids[4][1], centroids[6][1]] ]

    (m0, b0) = np.polyfit(diag_x[0], diag_y[0], 1)
    (m1, b1) = np.polyfit(diag_x[1], diag_y[1], 1)

    border_points = []
    for i in [0,2,6,8]:
        x_val = centroids[i][0] - centroids[4][0] + centroids[i][0]

        if i==0 or i==8: border_points.append((x_val, m0*x_val+b0))
        else: border_points.append((x_val, m1*x_val+b1))

    x = np.linspace(min(centroids[0][0], centroids[3][0], centroids[6][0]), max(centroids[2][0], centroids[5][0], centroids[8][0]))
    #plt.plot(x,m0*x+b0)
    #plt.plot(x,m1*x+b1)

    # for (x,y) in border_points:
    #     plt.plot(x,y,'*', color='black')


    (m_y,b_y) = np.polyfit([border_points[3][0], border_points[2][0]], [border_points[3][1], border_points[2][1]], 1) #y_offset
    (m_y2,b_y2) = np.polyfit([border_points[1][0], border_points[0][0]], [border_points[1][1], border_points[0][1]], 1)
    (m_x, b_x) = np.polyfit([border_points[3][1], border_points[1][1]], [border_points[3][0], border_points[1][0]], 1) #x_offset
    (m_x2, b_x2) = np.polyfit([border_points[2][1], border_points[0][1]], [border_points[2][0], border_points[0][0]], 1) #x_offset

    n_x = np.linspace(border_points[3][0], border_points[2][0], n+2)
    n_y = np.linspace(border_points[3][1], border_points[1][1], n+2)
   
    n_x = np.delete(n_x, -1)
    n_y = np.delete(n_y, -1)
    n_x = np.delete(n_x, 0)
    n_y = np.delete(n_y, 0)


    #n_cents = [(nx + ny*m_x, ny + ((nx*m_y + nx*m_y2)/2)) for (nx, ny) in product(n_x,n_y)]
    #n_cents = [(nx + (ny-border_points[3][1])*m_x, ny + nx*m_y) for (nx, ny) in product(n_x,n_y)]
    n_cents = []
    for nx, ny in product(n_x,n_y):
        x_offset = (ny*m_x+b_x-border_points[3][0] + ny*m_x2+b_x2-border_points[2][0])/2
        y_offset = (nx*m_y+b_y-border_points[3][1] + nx*m_y2+b_y2-border_points[1][1])/2

        n_cents.append((nx+x_offset, ny+y_offset))

    for (x,y) in n_cents:
        plt.plot(x,y,'*', color='purple', label="Recalibrated")



def plot_data(datadir, cent, lines=False):
    """
    Plot all data provided onto multiple subplots, optionally plot the calibrated centroids and grid divisions

    data: List of lists, each sublist will be plotted on its own subfigure
    centroids: List of lists, calibrated centroid positions. If provided, will plot mean centroid positions along with grid delineations

    """

    centroids = load_centroids(cent)

    data = []
    conf_thresh = .75

    for file in os.listdir(datadir):
        filename = os.fsdecode(file)
        
        if filename.endswith(".csv") and "filled" in filename and "centroids" not in filename:   #Error handling to dodge hidden files and subdirectories
            print(filename)
            read = []
            with open(datadir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
            
            data.append(read)

    n_columns = math.ceil(len(data)/2)
    plt.figure(1)
    n = 1
    
    figure, axes = plt.subplots(nrows=2, ncols=2)
    for run in data:
        d = np.array(run).T
        ax = plt.subplot(2,n_columns,n)

        if n==1: ax.title.set_text('Conversation')
        elif n==2: ax.title.set_text('Reading')
        elif n==3: ax.title.set_text('Simulated Seizure')
        elif n==4: ax.title.set_text('Watching TV')

        for (x, y) in zip(d[0], d[1]):
            plt.plot(x, y, '.') 
        if centroids != None:
            for (x,y) in centroids:
                plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y])))
            
            if lines:
                #vertical lines
                plt.plot([(centroids[0][0] + centroids[1][0])/2,(centroids[6][0] + centroids[7][0])/2], [(centroids[0][1] + centroids[1][1])/2,(centroids[6][1] + centroids[7][1])/2],'--', color='black')
                plt.plot([(centroids[1][0] + centroids[2][0])/2,(centroids[7][0] + centroids[8][0])/2], [(centroids[1][1] + centroids[2][1])/2,(centroids[7][1] + centroids[8][1])/2],'--', color='black')

                #horizontal lines
                plt.plot([(centroids[0][0] + centroids[3][0])/2,(centroids[2][0] + centroids[5][0])/2], [(centroids[0][1] + centroids[3][1])/2,(centroids[2][1] + centroids[5][1])/2],'--', color='black')
                plt.plot([(centroids[3][0] + centroids[6][0])/2,(centroids[5][0] + centroids[8][0])/2], [(centroids[3][1] + centroids[6][1])/2,(centroids[5][1] + centroids[8][1])/2],'--', color='black')
        n=n+1

    figure.tight_layout()
    plt.show()

def plot_all_cents(experdir, cent):
    """
    Plot all centroids for all subjects
    Draw a line between each adjacent centroid
    """
    plt.figure()
    i=0
    for dirName, subdirList, fileList in os.walk(experdir):
        
        if "subj" in dirName:
            centroids = load_centroids(dirName + "/" + cent)
            

            for (x,y) in centroids:
                plt.plot(x,y,'*', color=class_to_color(i))
            
            plt.plot([centroids[0][0],centroids[1][0]],[centroids[0][1],centroids[1][1]], color=class_to_color(i))
            plt.plot([centroids[0][0],centroids[3][0]],[centroids[0][1],centroids[3][1]], color=class_to_color(i))
            plt.plot([centroids[1][0],centroids[2][0]],[centroids[1][1],centroids[2][1]], color=class_to_color(i))
            plt.plot([centroids[1][0],centroids[4][0]],[centroids[1][1],centroids[4][1]], color=class_to_color(i))
            plt.plot([centroids[2][0],centroids[5][0]],[centroids[2][1],centroids[5][1]], color=class_to_color(i))
            plt.plot([centroids[3][0],centroids[4][0]],[centroids[3][1],centroids[4][1]], color=class_to_color(i))
            plt.plot([centroids[3][0],centroids[6][0]],[centroids[3][1],centroids[6][1]], color=class_to_color(i))
            plt.plot([centroids[4][0],centroids[5][0]],[centroids[4][1],centroids[5][1]], color=class_to_color(i))
            plt.plot([centroids[4][0],centroids[7][0]],[centroids[4][1],centroids[7][1]], color=class_to_color(i))
            plt.plot([centroids[5][0],centroids[8][0]],[centroids[5][1],centroids[8][1]], color=class_to_color(i))
            plt.plot([centroids[6][0],centroids[7][0]],[centroids[6][1],centroids[7][1]], color=class_to_color(i))
            plt.plot([centroids[7][0],centroids[8][0]],[centroids[7][1],centroids[8][1]], color=class_to_color(i))
            i+=1
        
    plt.title("All Subject Centroids")
    plt.savefig("figures/allcentroids_", dpi=600, bbox_inches="tight")



def plot_accel(data):
    """
    Plot accelerometry and gyroscopic data signatures from IMU
    """
    n = 0
    for run in data:
        d = np.array(run).T
        t = np.linspace(0,len(d[0]),len(d[0]))
        figure, axes = plt.subplots(nrows=2, ncols=3)
        figure.tight_layout()
        for i in range(1,7):
            plt.subplot(2,3,i)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            

            plt.plot(t, d[i+3])
            
            if i==1 and n==0:
                mid = -7
            elif i==2 and n==0:
                mid = -2.5
            elif i==3 and n==0:
                mid = 3.5

            if i==1 or i==2 or i==3 and n==0:
                
                ax.set_ylim([mid-9, mid+9])

            if i==4 or i==5 or i==6 and n==0:
                
                ax.set_ylim([-1000, 1000])
            
            if i==1 and n==1:
                mid = -10
            elif i==2 and n==1:
                mid = 0
            elif i==3 and n==1:
                mid = 5
            if i==1 or i==2 or i==3 and n==1:
                ax.set_ylim([mid-6, mid+6])
            if i==4 or i==5 or i==6 and n==1:
                ax.set_ylim([-2000, 2000])
        
        plt.savefig('figures/' + str(n) + ".png", dpi=600)
        n+=1
        plt.figure()



if __name__ == "__main__":
    plot_centroids(load_centroids("experiment/subj4/centroids_0.csv"), n=10)