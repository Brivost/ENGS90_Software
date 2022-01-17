# To run this script, you need to run: 
#       `pip install tk`
#       `pip install zmq msgpack==0.5.6`
#       `pip install matplotlib` 
#
# Ensure that pip has installed these packages to the PythonPath the IDE is running
# graphics.py must be findable by python

import ssl
import zmq
from graphics import *
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from cvd_pupillometry.pyplr.pupil import PupilCore
from cvd_pupillometry.pyplr.utils import unpack_data_pandas

def calibrate():
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    grid_w = screen_width/3
    grid_h = screen_height/3

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 3     #time for calibration at each point
    conf_thresh = .7   #confidence threshold

    win = GraphWin("Calibration", screen_width, screen_height)

    
    head = Text(Point(screen_width/2,screen_height/3), "Calibration Pt. 1").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Focus on each dot for 10 seconds" + '\n' + "Press any key once your eyes are focused on the dot" + '\n' + "Press any key to begin").draw(win)
    sub.setSize(20)

    win.getKey()
    head.undraw()
    sub.undraw()
    
#Connect to Pupil Core
    ctx = zmq.Context()
    pupil_remote = zmq.Socket(ctx, zmq.REQ)
    pupil_remote.connect('tcp://127.0.0.1:50020')

    p = PupilCore() # having issues here
    calibration = []
    traces = []

# Calibrate at the centroid of each grid
    for i in range(0,3):
        for j in range(0,3):
            #Move the point
            ball = Circle(Point(grid_w/2+ grid_w*j,grid_h/2+ grid_h*i), radius)
            ball.setFill(color)
            ball.setOutline(color)
            ball.draw(win)
            win.getKey()

            #Calibrate

            pgr_future = p.pupil_grabber(topic='pupil.0.3d', seconds=calibrate_t)
            data = pgr_future.result()
            calibration.append([[d[b'norm_pos'][0] for d in data], [d[b'norm_pos'][1] for d in data], [d[b'confidence'] for d in data]])
            
            
            # pupil_remote.send_string('R') #Begin recording
            # print(pupil_remote.recv_string())
            # time.sleep(calibrate_t)
            # pupil_remote.send_string('r') #Stop recording
            # print(pupil_remote.recv_string())

            ball.undraw()
    
    
    head = Text(Point(screen_width/2,screen_height/3), "Calibration Pt. 2").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Initially fixate on the dot as you did in Part One" + '\n' + "Press any key once your eyes are focused on the dot and it will begin to move" + '\n' + "Do your best to follow the dot with your eyes as closely as you can").draw(win)
    sub.setSize(20)
    win.getKey()
    head.undraw()
    sub.undraw()
    
    ball = Circle(Point(screen_width/2,0+radius), radius)
    ball.setFill(color)
    ball.setOutline(color)
    ball.draw(win)
    win.getKey()
    
    pupil_remote.send_string('R') #Begin recording
    print(pupil_remote.recv_string())

    for i in range(screen_height-radius):
        ball.move(0,1)
        time.sleep(3/(screen_height-radius))

    pupil_remote.send_string('r') #Stop recording
    print(pupil_remote.recv_string())

    ball.undraw()

    ball = Circle(Point(0+radius,screen_height/2), radius)
    ball.setFill(color)
    ball.setOutline(color)
    ball.draw(win)
    win.getKey()
    
    pupil_remote.send_string('R') #Begin recording
    print(pupil_remote.recv_string())

    for i in range(screen_width-radius):
        ball.move(1,0)
        time.sleep(3/(screen_width-radius))

    pupil_remote.send_string('r') #Stop recording
    print(pupil_remote.recv_string())

    ball.undraw()
    
    head = Text(Point(screen_width/2,screen_height/3), "Finished!").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Press any key to exit").draw(win)
    sub.setSize(20)
    
    win.getKey()
    win.close()

    #Process the data, plot, and return
    trimmed = []
    for grid in calibration:
        to_trim = np.array(grid[2]) >= conf_thresh
        trimmed.append([np.array(grid[0])[to_trim],np.array(grid[1])[to_trim], np.array(grid[2])[to_trim]])
    
    calibration = trimmed
    
    for point in calibration:
        plt.plot(point[0], point[1], '.')
    plt.show()
    
    return [(np.average(grid[0]), np.average(grid[1])) for grid in calibration] 


# Returns an array of closest centroids for points in a data array
def classify(centroids, data):

    classified = []

    for point in data:
        for centroid in centroids:
            distances = ((((point[0]-centroid[0])**2) + ((point[1]-centroid[1])**2))**0.5)
        classified.append(min(distances))
    
    return classified # need to convert to 0-8? 

# Check accuracy of calibration
def validate():
    #Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    grid_w = screen_width/3
    grid_h = screen_height/3

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 3     #time for calibration at each point
    conf_thresh = .7   #confidence threshold

    win = GraphWin("Calibration", screen_width, screen_height)

    
    head = Text(Point(screen_width/2,screen_height/3), "Validation Pt. 1").draw(win)
    head.setSize(30)
    head.setStyle('bold')
    sub = Text(Point(screen_width/2,screen_height/3+75), "Focus on each dot for 10 seconds" + '\n' + "Press any key once your eyes are focused on the dot" + '\n' + "Press any key to begin").draw(win)
    sub.setSize(20)

if __name__ == "__main__":
    centroids = calibrate()
