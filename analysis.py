# To run this script, you need to install: 
#       `pip install matplotlib` 
#       `pip install sklearn
#
# Ensure that pip has installed these packages to the PythonPath the IDE is running

import csv
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split



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
        plt.plot(d[1], d[2], '.')
        if centroids != None:
            for (x,y) in centroids:
                plt.plot(x,y,'*')
        n=n+1
    
    plt.show()



def separate(features, labels):
    """
    Run an LDA on the provided features and labels
    """
    inds = set(random.sample(list(range(len(a))), int(frac*len(a))))

def feature_extraction(data, specific=False):
    """
    Extract features from raw data

    data: List of lists, each sublist is a run
    specific: If True, will return exact eye positions and confidence values in feature lists

    Returns: List of lists, of relevant features 
    """


if __name__ == "__main__":
    (data, centroids) = load('data/')
    plot_data(data, centroids)