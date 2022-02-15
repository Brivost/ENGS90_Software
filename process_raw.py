
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os


def classify(centroids, data, conf_thresh):
    """ Classifies pupil x,y position in data matrix into grid number
        Classification is simply determined by nearest calibrated point
        Returns data vector with classified eye position appended

        centroids: list of calibrated centroids
        data: eye_x, eye_y, eye_conf, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
        
    """
    labeled = []
    
    for sample in data:
        value = 9 #Classify as 9 if the sample does not meet the confidence threshold
        if sample[2] >= conf_thresh: 
            distances = [math.sqrt((sample[0] - point[0])**2 + (sample[1] - point[1])**2) for point in centroids]
            value = distances.index(min(distances))
        labeled.append(np.insert(sample, 0, value))
        
    return labeled

def process_all(experdir, centroid="centroids_0.csv", max_feat=False):
    for dirName, subdirList, fileList in os.walk(experdir):
        if "subj" in dirName:
            process(dirName + "/", centroid, dirName + "/", max_feat)


def process(rawdir, cent, outdir, max_feat=False):

    data = []
    centroids = []
    conf_thresh = .75


    with open(rawdir + cent, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            centroids.append([float(x) for x in row[0].split(',')])
        
    
    for file in os.listdir(rawdir):

        filename = os.fsdecode(file)
        if filename.endswith(".csv") and "centroid" not in filename and "data" not in filename:   #Error handling to dodge hidden files and subdirectories
            read = []
            with open(rawdir + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
    
                    read.append([float(x) for x in row[0].split(',')])
                    
            read = classify(centroids, np.array(read).astype(np.float), conf_thresh)
            data.append(read)
            
            with open(outdir + "data_" + filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for sample in read:
                    if not max_feat: writer.writerow([sample[i] for i in [0, 4,5,6,7,8,9]])
                    else: writer.writerow(sample)
                

    return (data, centroids)



def plot_data(data, centroids=None, lines=False):
    """
    Plot all data provided onto multiple subplots, optionally plot the calibrated centroids and grid divisions

    data: List of lists, each sublist will be plotted on its own subfigure
    centroids: List of lists, calibrated centroid positions. If provided, will plot mean centroid positions along with grid delineations

    """

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

        for (c, x, y) in zip(d[0], d[1], d[2]):
            plt.plot(x, y, '.', color=class_to_color(c)) 
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

if __name__ == "__main__":

    process_all("experiment/")
