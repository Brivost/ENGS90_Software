# To run this script, you need to install: 
#       `pip install matplotlib` 
#       `pip install sklearn
#
# Ensure that pip has installed these packages to the PythonPath the IDE is running

import csv
import os
import math
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics

def class_to_color(c):
    if c==0: return 'red'
    elif c==1: return 'pink'
    elif c==2: return 'orange'
    elif c==3: return 'yellow'
    elif c==4: return 'green'
    elif c==5: return 'blue'
    elif c==6: return 'brown'
    elif c==7: return 'purple'
    elif c==8: return 'black'

def load(datadir):
    """ 
    Loads in data from specified directory
    
    datadir: string, path to csv data

    Returns: (List of lists, list of lists), sublists are data from each .csv file in the specified directory, second list is centroids
    """
    data = []
    centroids = []
    for file in os.listdir(datadir):
        filename = os.fsdecode(file)

        if filename.endswith(".csv"):   #Error handling to dodge hidden files and subdirectories
            read = []
            with open(datadir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
            if filename != 'centroids.csv': data.append(read)
            else: centroids = read

    return (data, centroids)

def plot_data(data, centroids=None):
    """
    Plot all data provided onto multiple subplots, optionally plot the calibrated centroids and grid divisions

    data: List of lists, each sublist will be plotted on its own subfigure
    centroids: List of lists, calibrated centroid positions. If provided, will plot mean centroid positions along with grid delineations

    """

    n_columns = math.ceil(len(data)/2)
    plt.figure(1)
    n = 1
    for run in data:
        d = np.array(run).T
        plt.subplot(2,n_columns,n)
        for (c, x, y) in zip(d[0], d[1], d[2]):
            plt.plot(x, y, '.', color=class_to_color(c)) 
        if centroids != None:
            for (x,y) in centroids:
                plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y])))
            
            #vertical lines
            plt.plot([(centroids[0][0] + centroids[1][0])/2,(centroids[6][0] + centroids[7][0])/2], [(centroids[0][1] + centroids[1][1])/2,(centroids[6][1] + centroids[7][1])/2],'--', color='black')
            plt.plot([(centroids[1][0] + centroids[2][0])/2,(centroids[7][0] + centroids[8][0])/2], [(centroids[1][1] + centroids[2][1])/2,(centroids[7][1] + centroids[8][1])/2],'--', color='black')

            #horizontal lines
            plt.plot([(centroids[0][0] + centroids[3][0])/2,(centroids[2][0] + centroids[5][0])/2], [(centroids[0][1] + centroids[3][1])/2,(centroids[2][1] + centroids[5][1])/2],'--', color='black')
            plt.plot([(centroids[3][0] + centroids[6][0])/2,(centroids[5][0] + centroids[8][0])/2], [(centroids[3][1] + centroids[6][1])/2,(centroids[5][1] + centroids[8][1])/2],'--', color='black')
        n=n+1

    plt.show()



def separate(features, labels):
    """
    Run an LDA on the provided features and labels
    """
    train_feat, test_feat, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_feat, train_labels)
    preds = clf.predict(test_feat)
    print(preds)
    print("****")
    print(test_labels)
    fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)


def feature_extraction(data, labs, outdir):
    """
    Extract features from raw data

    data: List of lists, each sublist is a run
    labs: List of len(data), a label for each run

    Returns: List of lists, of relevant features 
    """
    epoch_size = 130
    feats = []
    labels = []
    for (run, l) in zip(data, labs):
        n = math.floor(len(run)/(epoch_size/2))
        l = (n-1)*[l]
        labels += l
        for i in range(1,n):
            f = []

            splice = np.array(run[(int)((i-1)*(epoch_size/2)):(int)((i*epoch_size/2)+epoch_size/2)])
            classified = splice[:,0]

            #Eye Tracking
            f.append(statistics.mean(classified))                                                                                   #Average class
            f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
            f.append((np.diff(classified)!=0).sum())                                                                                #Number of times class changes
            f.append((classified==9).sum())                                                                                         #Number of poor confidence values

            #Head Tracking

            f.append(statistics.mean([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))
            f.append(statistics.mean([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))  
            
            f.append(statistics.stdev([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))         #Standard Deviation of acceleration
            f.append(statistics.stdev([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))         #Standard Deviation magnitude of gyroscope
            
            f.append(max([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))                      #Maximum magnitude of acceleration
            f.append(max([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))                      #Maximum magnitude of gyroscope

            f.append(min([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))                      #Minmum magnitude of acceleration
            f.append(min([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))                      #Minimum magnitude of gyroscope


            #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))))   #Average Fourier transform of acceleration magnitude
            #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))))   #Average Fourier transform of gyroscope magnitude
            
            feats.append(f)

 
    with open(outdir + "sub/features.csv", 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(feats)):
            writer.writerow(feats[i] + [labels[i]])

    return (feats, labels)




if __name__ == "__main__":

    (data, centroids) = load('readwatch/')
    (feat, lab) = feature_extraction(data, [0,1,1], 'readwatch/')
    print(len(feat))
    separate(feat, lab)


    #plot_data(data, centroids)
