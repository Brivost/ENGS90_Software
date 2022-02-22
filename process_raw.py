
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
from itertools import product
#from analysis import class_to_color, separate
from scipy.spatial import distance

def extract_hvcents(experdir):
    """
    Using 3x3 centroids, write out 1x3 and 3x1 centroids
    """

    print("Writing horizontal and vertical centroid files to the experiment directory")
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

    n_cents = []
    # for ny, nx in product(reversed(n_y),n_x):
    #     x_offset = (ny*m_x+b_x-border_points[3][0] + ny*m_x2+b_x2-border_points[2][0])/2
    #     y_offset = (nx*m_y+b_y-border_points[3][1] + nx*m_y2+b_y2-border_points[1][1])/2

    #     n_cents.append((nx+x_offset, ny+y_offset))
    for nx, ny in product(n_x,n_y):
        x_offset = (ny*m_x+b_x-border_points[3][0] + ny*m_x2+b_x2-border_points[2][0])/2
        y_offset = (nx*m_y+b_y-border_points[3][1] + nx*m_y2+b_y2-border_points[1][1])/2

        n_cents.append((nx+x_offset, ny+y_offset))

    reorder_ncent = []
    for j in reversed(range(0,n)):
        for i in reversed(range(0,n)):
            reorder_ncent.append(n_cents[i*n+j])


    return reorder_ncent

def extract_ncent(experdir, n):
    print("Writing horizontal and vertical centroid files to the experiment directory")
    for dirName, subdirList, fileList in os.walk(experdir):    
        if "subj" in dirName:
            print(dirName)
            centroids = calc_ncent(load_centroids(dirName + "/centroids_0.csv"),n)
            with open(dirName + "/centroids_" + str(n) + ".csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for point in centroids:
                    writer.writerow(point)
 



def classify(centroids, data, n):
    """ Classifies pupil x,y position in data matrix into grid number
        Classification is simply determined by nearest calibrated point
        Returns data vector with classified eye position appended

        centroids: list of calibrated centroids
        data: eye_x, eye_y, eye_conf, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
        
    """
    labeled = []
    conf_thresh = 0
    for sample in data:
        value = n*n #Classify as 9 if the sample does not meet the confidence threshold
        if sample[2] >= conf_thresh: 
            distances = [math.sqrt((sample[0] - point[0])**2 + (sample[1] - point[1])**2) for point in centroids]
            value = distances.index(min(distances))
        labeled.append(np.insert(sample, 0, value))
        
    return labeled

def process_all(experdir, centroid="centroids_0.csv", grid=3,max_feat=False):
    
    n = 0.0
    total = 0.0
    for dirName, subdirList, fileList in os.walk(experdir):
        if "subj" in dirName:
            (n, total) = process(dirName + "/", centroid, dirName + "/", n, total, grid, max_feat)
    print("Low Confidence: " + str(n / total))


def process(rawdir, cent, outdir, n,t, grid, max_feat=False):

    data = []
    centroids = []
    curr = n
    total = t


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
                    
            read = classify(centroids, np.array(read).astype(np.float), grid)
            try:
                r = np.array(read)[:,0]
            except IndexError:
                pass
            
            
            curr += r[r==grid*grid].shape[0]
            total+=len(read)
            data.append(read)
            
            with open(outdir + "data_" + filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for sample in read:
                    if not max_feat: writer.writerow([sample[i] for i in [0, 4,5,6,7,8,9]])
                    else: writer.writerow(sample)
                

    return (curr, total)

def load_centroids(cent):
    """
    Load in centroids 
    """
    centroids = []
    with open(cent, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            centroids.append([float(x) for x in row[0].split(',')])
    return centroids

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
        
        if filename.endswith(".csv") and "data" not in filename and "centroids" not in filename:   #Error handling to dodge hidden files and subdirectories
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
        
        if filename.endswith(".csv") and "data" not in filename and "centroid" not in filename:   #Error handling to dodge hidden files and subdirectories
            print(filename)
            read = []
            with open(datadir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
                        
                i = 0
                if c: 
                    read = classify(centroids, np.array(read).astype(np.float), conf_thresh)
                    i = 1
                for add in read:
                    if math.isnan(add[0+i]):
                        add[0+i] = 9
                        add[1+i] = 9
                        add[2+i] = 0

            
            data.append(read)
            
    
 
    return data


def full_feature_extraction(data, labs, outdir, et=.25, c=False):
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
        n = math.floor(len(run)/(epoch_size/2))

        num_sample = 0
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
                f.append(statistics.mean(x))
                f.append(statistics.mean(y)) 
                f.append(max(x))
                f.append(min(x))
                f.append(max(y))
                f.append(min(y))
                f.append(max(x) - min(x))
                f.append(max(y) - min(y))                                                                                  
                f.append(np.amax(dists))
                f.append(np.amin(dists))
                f.append(np.amax(dists) - np.amin(dists))                                                                                                                                                                     #Number of times class changes
                f.append(np.matrix(dists).mean())                                                                               
                f.append(statistics.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i+1])) for i in range(len(coords)-1)]))

                 
                f.append(statistics.mean(conf))
                f.append(statistics.stdev(conf))
                f.append((splice[:,0+offset]==9).sum())       

                if c:
                    classified = splice[:,0]
                    f.append(statistics.mean(classified))                                                                                   #Average class
                    f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                    f.append((np.diff(classified)!=0).sum())  
                    f.append((splice[:,0+offset]==9).sum())                                                                                #Number of times class changes             
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
                
                f.append(accel_max - accel_min)
                f.append(gyro_max - gyro_min)      


                #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))))   #Average Fourier transform of acceleration magnitude
                #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))))   #Average Fourier transform of gyroscope magnitude
                
                feats.append(f)
                num_sample+=1
                
                # if max(f) > 100000: 
                #     print("HERE")
                #     print(f)
                # array_sum = np.sum(np.array(f))
                # array_has_nan = np.isnan(array_sum)
                # if array_has_nan:
                #     print(f)
        
        l = (num_sample)*[l]
        labels += l
    print(len(feats))
    return(feats,labels)





if __name__ == "__main__":
    #extract_hvcents("experiment/")
    extract_ncent("experiment", 3)
    data = process_all("experiment/", centroid="centroids_3.csv")
    
    #data = load_all("experiment/", c=False,cent='/centroids_2x2.csv')
    #outdir = "figures/3x3Analysis/FullFeature_woClass/"

    """
    #Seizure vs Technology
    (feat, lab) = full_feature_extraction([data[0], data[1]], [1,0], 'experiment/features/', c=False)
    separate(feat, lab, outdir, title="Seizure vs Technology", n=0)

    #Seizure vs Eating
    (feat, lab) = full_feature_extraction([data[0], data[2]], [1,0], 'experiment/features/', c=False)
    separate(feat, lab, outdir, title="Seizure vs Eating", n=1)

    #Seizure vs Coversation
    (feat, lab) = full_feature_extraction([data[0], data[3]], [1,0], 'experiment/features/', c=False)
    separate(feat, lab, outdir, title="Seizure vs Conversation", n=2)

    #Seizure vs Non-Seizure
    (feat, lab) = full_feature_extraction(data, [1,0,0,0], 'experiment/features/',c=False)
    separate(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)
    
    #(feat, lab) = full_feature_extraction(data, [1,0,0,0], 'experiment/features/',c=True)
    #separate(feat, lab)

    #plot_data("experiment/subj1/", "experiment/subj1/centroids_0.csv")
    """
  