import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import statistics
from itertools import product,combinations
#from analysis import class_to_color, separate
from scipy.spatial import distance
import argparse

def extract_hvcents(experdir):
    """
    Using 3x3 centroids, write out 1x3 and 3x1 centroids
    """
    for dirName, subdirList, fileList in os.walk(experdir):    
        if "subj" in dirName:
            print(dirName)
            centroids = load_centroids(dirName + "/centroids_0.csv")
            with open(dirName + "/centroids_hor.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for point in [centroids[3],centroids[4],centroids[5]]:
                    writer.writerow(point)
            with open(dirName + "/centroids_ver.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for point in [centroids[1],centroids[4],centroids[7]]:
                    writer.writerow(point)


def calc_ncent(centroids, n):
    """
    Compute an NxN centroid grid from a 3x3 centroid grid

    centroids: list of centroids corresponding to a 3x3 grid
    """
    diag_x = [ [centroids[0][0] , centroids[4][0], centroids[8][0]], [centroids[2][0], centroids[4][0], centroids[6][0]] ]
    diag_y = [ [centroids[0][1] , centroids[4][1], centroids[8][1]], [centroids[2][1], centroids[4][1], centroids[6][1]] ]

    #Fit lines along the diagonals of the grid
    (m0, b0) = np.polyfit(diag_x[0], diag_y[0], 1)
    (m1, b1) = np.polyfit(diag_x[1], diag_y[1], 1)

    #Compute the four corner points defining the grid
    border_points = []
    for i in [0,2,6,8]:
        x_val = centroids[i][0] - centroids[4][0] + centroids[i][0]

        if i==0 or i==8: border_points.append((x_val, m0*x_val+b0))
        else: border_points.append((x_val, m1*x_val+b1))

    x = np.linspace(min(centroids[0][0], centroids[3][0], centroids[6][0]), max(centroids[2][0], centroids[5][0], centroids[8][0]))

    #fit lines along the edges of the border
    (m_y,b_y) = np.polyfit([border_points[3][0], border_points[2][0]], [border_points[3][1], border_points[2][1]], 1) #y_offset
    (m_y2,b_y2) = np.polyfit([border_points[1][0], border_points[0][0]], [border_points[1][1], border_points[0][1]], 1)
    (m_x, b_x) = np.polyfit([border_points[3][1], border_points[1][1]], [border_points[3][0], border_points[1][0]], 1) #x_offset
    (m_x2, b_x2) = np.polyfit([border_points[2][1], border_points[0][1]], [border_points[2][0], border_points[0][0]], 1) #x_offset

    #evenly space N points along the border lines
    n_x = np.linspace(border_points[3][0], border_points[2][0], n+2)
    n_y = np.linspace(border_points[3][1], border_points[1][1], n+2)
   
   #trim off the end points, corresponding to the edge of the grid
    n_x = np.delete(n_x, -1)
    n_y = np.delete(n_y, -1)
    n_x = np.delete(n_x, 0)
    n_y = np.delete(n_y, 0)

    n_cents = []
    
    #space points with averaged offset from border liens
    for nx, ny in product(n_x,n_y):
        x_offset = (ny*m_x+b_x-border_points[3][0] + ny*m_x2+b_x2-border_points[2][0])/2
        y_offset = (nx*m_y+b_y-border_points[3][1] + nx*m_y2+b_y2-border_points[1][1])/2

        n_cents.append((nx+x_offset, ny+y_offset))

    #reorder the centroids into the same way the 3x3 were ordered
    reorder_ncent = []
    for j in reversed(range(0,n)):
        for i in reversed(range(0,n)):
            reorder_ncent.append(n_cents[i*n+j])


    return reorder_ncent

def extract_ncent(experdir, n):
    """
    Write out a csv file containing coordinates for an NxN centroid grid
    """
    for dirName, subdirList, fileList in os.walk(experdir):    
        if "subj" in dirName:
            print(dirName)
            centroids = calc_ncent(load_centroids(dirName + "/centroids_0.csv"),n)
            with open(dirName + "/centroids_" + str(n) + ".csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for point in centroids:
                    writer.writerow(point)
 



def classify(centroids, data, n, conf_t=0):
    """ Classifies pupil x,y position in data matrix into grid number
        Classification is simply determined by nearest calibrated point
        Returns data vector with classified eye position appended

        centroids: list of calibrated centroids
        data: eye_x, eye_y, eye_conf, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
        
    """
    labeled = []
    conf_thresh = conf_t
    for sample in data:
        value = n*n #Classify as 9 if the sample does not meet the confidence threshold
        if sample[2] >= conf_thresh: 
            distances = [math.sqrt((sample[0] - point[0])**2 + (sample[1] - point[1])**2) for point in centroids]
            value = distances.index(min(distances))
        labeled.append(np.insert(sample, 0, value))
        
    return labeled

def process_all(experdir, centroid="centroids_0.csv", grid=3,limit=None,max_feat=False,conf_t=0):
    """
    Write out .csv files with classified eye position
    Calculate number of low confidence samples in the dataset
    """
    n = 0.0
    total = 0.0
    for dirName, subdirList, fileList in os.walk(experdir):
        if "subj" in dirName:
            #(n, total) = process(dirName + "/", centroid, dirName + "/", n, total, grid, max_feat)
            (n,total) = fill_gaps(dirName + "/", centroid, dirName + "/", n, total, grid, limit=limit,conf_t=conf_t)
    print("Low Confidence: " + str(n / total))


def process(rawdir, cent, outdir, n,t, grid, max_feat=False):
    """
    Process .csv files in subject data directory, classifying eye position and writing out a new .csv
    """
    data = []
    centroids = []
    curr = n
    total = t

    #open specified centroid file
    with open(rawdir + cent, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            centroids.append([float(x) for x in row[0].split(',')])
    
    
    for file in os.listdir(rawdir):
        #Read in proper data file, process line by line
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and "centroid" not in filename and "data" not in filename:   #Error handling to dodge hidden files and subdirectories
            read = []
            with open(rawdir + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
                    
            read = classify(centroids, np.array(read).astype(np.float), grid)
            try:
                r = np.array(read)[:,0]
            except IndexError:
                pass
            
            #Keep track of number of low-confidence samples
            curr += r[r==grid*grid].shape[0]
            total+=len(read)
            data.append(read)
            
            #Write out processed data file
            with open(outdir + "data_" + filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for sample in read:
                    if not max_feat: writer.writerow([sample[i] for i in [0, 4,5,6,7,8,9]])
                    else: writer.writerow(sample)
                

    return (curr, total)


def fill_gaps(rawdir, cent, outdir, n,t, grid, limit=None, max_feat=False,conf_t=None):
    """
    Similar to process, but include K-Fill Data Imputation to reduce low confidence samples
    """
    data = []
    centroids = []
    conf_thresh = 0
    curr = n
    total = t

    #Open specified centroid file
    with open(rawdir + cent, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            centroids.append([float(x) for x in row[0].split(',')])
    
    for file in os.listdir(rawdir):
        filename = os.fsdecode(file)
        
        if filename.endswith(".csv") and "data" not in filename and "centroid" not in filename and "filled" not in filename:   #Error handling to dodge hidden files and subdirectories
            print(filename)
            read = []
            #Read in samples
            with open(rawdir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
            
            #Use pandas to perform K-Fill data imputation
            read = np.array(read).T      
            x = pd.Series(read[0])
            x = x.interpolate(method='nearest',limit_direction='both', limit=limit)

            y = pd.Series(read[1])
            y = y.interpolate(method='nearest',limit_direction='both', limit=limit)
            
            conf = pd.Series(read[2])
            conf = conf.interpolate(method='nearest',limit_direction='both', limit=limit)           

            read = np.array([x.tolist(),y.tolist(),conf.tolist(),read[3],read[4],read[5],read[6],read[7],read[8]]).T

            #Classify with imputation
            read = classify(centroids, read.astype(np.float), grid, conf_t)
            

            try:
                r = np.array(read)[:,0]
            except IndexError:
                pass
            
            #Keep track of low-confidence samples
            curr += r[r==grid*grid].shape[0]
            total+=len(read)
            data.append(read)
            
            #Write out classified file
            with open(outdir + "data_" + filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for sample in read:
                    if not max_feat: writer.writerow([sample[i] for i in [0,4,5,6,7,8,9]])
                    else: writer.writerow(sample)
                

    return (curr, total)
            
    
 
def load_centroids(cent):
    """
    Load in centroids

    cent: specified pathname to centroids
    """
    centroids = []
    with open(cent, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            centroids.append([float(x) for x in row[0].split(',')])
    return centroids



def load_all(experdir, c=False,cent=None):
    """
    Loads all the data from the experiment. Specific to the way the output files are saved

    experdir: String, path to experiment directory

    Returns [ [[conversation_0],...,[conversation_n]], [[eating_0],...,[eating_n]], [[technology_0],..,[technology_n]], [[seizure_0],..,[seizure_n]] ]
    """
    data = [ [], [], [], []] 
    for dirName, subdirList, fileList in os.walk(experdir):
        if "subj" in dirName:
            print(dirName)
            read = load(dirName,c,cent)
            #sort into specified order
            for i in range(len(read)):
                if i == 0: j = 1
                elif i == 1: j = 3
                elif i == 2: j = 0
                elif i == 3: j = 2
                else: j = 0
                
                data[j].extend(read[i])
               
    return data
        

def load(datadir, c, cent):
    """ 
    Loads in data from specified directory
    
    datadir: string, path to csv data


    Returns: List of lists, sublists are data from each .csv file in the specified directory
    """
    data = []
    conf_thresh = 0

    if c: centroids = load_centroids(datadir + cent)
    
    for file in os.listdir(datadir):
        filename = os.fsdecode(file)
        
        if filename.endswith(".csv") and "filled" in filename and "centroid" not in filename:   #Error handling to dodge hidden files and subdirectories
            print(filename)
            read = []
            with open(datadir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
                        
                i = 1

                for add in read:
                    if math.isnan(add[0+i]):
                        add[0+i] = 9
                        add[1+i] = 9
                        add[2+i] = 0

            
            data.append(read)
            
    
 
    return data


def full_feature_extraction(data, labs, outdir, et=.05, c=False):
    """
    Extract features from raw data

    data: List of lists, each sublist is a run
    labs: List of len(data), a label for each run

    Returns: List of lists, of relevant features 
    """
    epoch_size = 130
    feats = []
    labels = []
    epoch_thresh = et

    if c: offset = 1
    else: offset = 0

    for (run, l) in zip(data, labs):
        #Calculate number of epochs
        n = math.floor(len(run)/(epoch_size/2))

        num_sample = 0
        #Split into epochs
        for i in range(1,n):
            f = []

            splice = np.array(run[(int)((i-1)*(epoch_size/2)):(int)((i*epoch_size/2)+epoch_size/2)])

            
            if (splice[:,0+offset][splice[:,0+offset]==9].shape[0] / len(splice[:,0+offset])) <= epoch_thresh:

                trimmed = np.delete(splice, np.where((splice[:, 0+offset] == 9))[0], axis=0)
                 
                x = trimmed[:,0+offset]
                y = trimmed[:,1+offset]
                conf = trimmed[:,2+offset]
                coords = [[xp,yp] for (xp,yp) in zip(x,y)]

                dists = distance.cdist(np.vstack((x,y)), np.vstack((x,y)), 'euclidean')

                #Eye Tracking
                f.append(statistics.mean(x))    #Average x
                f.append(statistics.mean(y))    #Average y
                f.append(max(x))                #Max X
                f.append(min(x))                #Min X
                f.append(max(y))                #Max Y
                f.append(min(y))                #Min Y
                f.append(max(x) - min(x))       #Range X
                f.append(max(y) - min(y))       #Range Y                                                                   
                f.append(np.amax(dists))        #Max distance between points
                f.append(np.amin(dists))        #Min distance between points
                f.append(np.amax(dists) - np.amin(dists))                   #Range distance between points                                                                                                                                                  #Number of times class changes
                f.append(np.matrix(dists).mean())                           #Average distance between all points                      
                f.append(statistics.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i+1])) for i in range(len(coords)-1)])) #Average distance between adjacent samples

                 
                f.append(statistics.mean(conf))     #Average confidence
                f.append(statistics.stdev(conf))    #Standard Deviation Confidence
                

                if c:
                    classified = splice[:,0]
                    f.append(statistics.mean(classified))                                                                                   #Average class
                    f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                    f.append((np.diff(classified)!=0).sum())                                                                                #Number of class changes
                    f.append((splice[:,0+offset]==9).sum())                                                                                 #Number of failed samples           
                    f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                    f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right

                #Head Tracking
                accel_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,3+offset]**2,splice[:,4+offset]**2,splice[:,5+offset]**2)]
                gyro_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,6+offset]**2,splice[:,7+offset]**2,splice[:,8+offset]**2)]

                f.append(statistics.mean(accel_mag))
                f.append(statistics.mean(gyro_mag))  
                
                f.append(statistics.stdev(accel_mag))         #Standard Deviation of acceleration
                f.append(statistics.stdev(gyro_mag))         #Standard Deviation magnitude of gyroscope
                
                accel_max = max(accel_mag)
                gyro_max = max(gyro_mag)

                accel_min = min(accel_mag)
                gyro_min = min(gyro_mag)

                f.append(accel_max)                      #Maximum magnitude of acceleration
                f.append(gyro_max)                      #Maximum magnitude of gyroscope

                f.append(accel_min)                      #Minmum magnitude of acceleration
                f.append(gyro_min)                      #Minimum magnitude of gyroscope
                
                f.append(accel_max - accel_min)         #Range accelerometer
                f.append(gyro_max - gyro_min)           #Range gyroscope

                feats.append(f)
                num_sample+=1

        
        l = (num_sample)*[l]
        labels += l
    print(len(feats))
    return(feats,labels)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", metavar=("Experiment directory"))
    parser.add_argument("-c", metavar=("Centroid file"))
    args = parser.parse_args()

    process_all(args.e, args.c,grid=3,limit=5)



    
  