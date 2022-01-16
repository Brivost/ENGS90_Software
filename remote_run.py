# To run this script, you need to run: 
#       `pip install tk`
#       `pip install zmq msgpack==0.5.6`
# 
# Ensure that pip has installed these packages to the PythonPath the IDE is running
# graphics.py must be findable by python

import zmq
from graphics import *
import tkinter as tk


if __name__ == "__main__":

#Set constants   
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    grid_w = screen_width/3
    grid_h = screen_height/3

    radius = 30         #dot properties
    color = 'black'
    calibrate_t = 10    #time for calibration at each point

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
            pupil_remote.send_string('R') #Begin recording
            print(pupil_remote.recv_string())
            time.sleep(calibrate_t)
            pupil_remote.send_string('r') #Stop recording
            print(pupil_remote.recv_string())

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