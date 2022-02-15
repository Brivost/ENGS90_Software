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
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

def class_to_color(c):
    if   c==0: return 'red'
    elif c==1: return 'pink'
    elif c==2: return 'orange'
    elif c==3: return 'yellow'
    elif c==4: return 'green'
    elif c==5: return 'blue'
    elif c==6: return 'brown'
    elif c==7: return 'purple'
    elif c==8: return 'black'


def load_all(experdir, merge=True):
    """
    Loads all the data from the experiment. Specific to the way the output files are saved

    experdir: String, path to experiment directory

    Returns [ [[conversation_0],...,[conversation_n]], [[eating_0],...,[eating_n]], [[technology_0],..,[technology_n]], [[seizure_0],..,[seizure_n]] ]
    """
    data = [ [], [], [], []] 
    for dirName, subdirList, fileList in os.walk(experdir):
        if "subj" in dirName:
            print(dirName)
            read = load(dirName)

            for i in range(len(read)):
                if i > 0:
                    data[i-1].extend(read[i])
                else:
                    data[i].extend(read[i])
    return data
        

def load(datadir):
    """ 
    Loads in data from specified directory
    
    datadir: string, path to csv data


    Returns: List of lists, sublists are data from each .csv file in the specified directory
    """
    data = []
    conf_thresh = .75

    for file in os.listdir(datadir):
        filename = os.fsdecode(file)
        
        if filename.endswith(".csv") and "data" in filename:   #Error handling to dodge hidden files and subdirectories
            print(filename)
            read = []
            with open(datadir + '/' + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    read.append([float(x) for x in row[0].split(',')])
            
            data.append(read)
            
    
 
    return data

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
            """
            if i==1: 
                plt.title("X Axis")
                plt.ylabel('Accelerometer (m/s^2)')
            elif i==2: plt.title("Y Axis") 
            elif i==3: plt.title("Z Axis") 
            elif i==4: plt.ylabel("Gyroscope (deg/s)")
            """
            
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
        
        plt.savefig('figures/' + str(n) + ".png")
        n+=1
        plt.figure()






def separate(features, labels):
    """
    Run an LDA on the provided features and labels
    """
    train_feat, test_feat, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
    clf = LinearDiscriminantAnalysis()
    #clf = KNeighborsClassifier(n_neighbors=5)
    
    #clf.fit(train_feat, train_labels)
    
    clf.fit_transform(train_feat, train_labels)
    clf.transform(test_feat)
    
    preds = clf.predict(test_feat)

    
    probs = clf.predict_proba(test_feat)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    auc = metrics.roc_auc_score(test_labels, probs)
    f1 = f1_score(test_labels, preds, average='macro')
    print('AUC: %.3f' % auc)
    print('F1: %.3f', % f1)


    #metrics.plot_roc_curve(clf, test_feat, test_labels)
    

    #plt.figure()
    #importance = 10*clf.coef_[0]
    #print(importance)
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    #full = ['Average Eye', '#Unique', '#Changes', '#Low-Conf', '%TB', '%LR' 'Avg Accel Mag', 'Avg Gyro Mag', 'Std Accel Mag', 'Std Gyro Mag', 'Accel Max', 'Gyro Max', 'Accel Min', 'Gyro Min']
    #justeye = ['Average Eye Class', '#Unique', '#Class Changes', '#Low-Confidence']
    #plt.bar(full, importance)
    #plt.xticks(rotation=90, fontsize=8)
    #plt.show()

    #plot_histo(features, labels, importance)
    auc = 0
    return (auc,f1)


def plot_histo(features, labels, coef):
    mapped = []
    # for feat in features:
    #     for (c, f) in zip(coef, feat):
    #         print(c*f)

    norm_coef = coef / np.linalg.norm(np.array(coef))
    features = np.array(features).dot(norm_coef.T)

    

    feat_1 = []
    feat_0 = []

    for (feat, l) in zip(features,labels):
        if l==0: feat_1.append(feat)
        else: feat_0.append(feat)
            
    #mapped_1 = [sum([c*f for (c,f) in zip(coef, feat)]) for feat in feat_1]
    #mapped_0 = [sum([c*f for (c,f) in zip(coef, feat)]) for feat in feat_0]

    plt.figure()
    plt.hist(feat_1, bins=50, color='blue', stacked=True, alpha=0.8, ec='black', density=True)
    plt.hist(feat_0, bins=50, color='red', stacked=True, alpha=0.8, ec='black', density=True)

    xmin, xmax = plt.xlim()
    mu1, std1 = norm.fit(feat_1)
    mu0, std0 = norm.fit(feat_0)

    x = np.linspace(xmin, xmax, 100)
    p1 = norm.pdf(x, mu1, std1)
    p0 = norm.pdf(x, mu0, std0)

    plt.plot(x, p1, color='blue')
    plt.plot(x, p0, color='red')


    plt.show()

def feature_extraction(data, labs, outdir, et=.45):
    """
    Extract features from raw data

    data: List of lists, each sublist is a run
    labs: List of len(data), a label for each run

    Returns: List of lists, of relevant features 
    """
    epoch_size = 260
    feats = []
    labels = []
    epoch_thresh = et

    for (run, l) in zip(data, labs):
        n = math.floor(len(run)/(epoch_size/2))

        num_sample = 0
        for i in range(1,n):
            f = []

            splice = np.array(run[(int)((i-1)*(epoch_size/2)):(int)((i*epoch_size/2)+epoch_size/2)])
            classified = splice[:,0]
            if (classified[classified==9].shape[0] / len(classified)) <= epoch_thresh:
                #Eye Tracking
                f.append(statistics.mean(classified))                                                                                   #Average class
                f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                f.append((np.diff(classified)!=0).sum())                                                                                #Number of times class changes
                f.append((classified==9).sum())                                                                                         #Number of poor confidence values
                f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right


                #Head Tracking
                
                f.append(statistics.mean([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,1]**2,splice[:,2]**2,splice[:,3]**2)]))
                f.append(statistics.mean([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))  
                
                f.append(statistics.stdev([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,1]**2,splice[:,2]**2,splice[:,3]**2)]))         #Standard Deviation of acceleration
                f.append(statistics.stdev([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))         #Standard Deviation magnitude of gyroscope
                
                f.append(max([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,1]**2,splice[:,2]**2,splice[:,3]**2)]))                      #Maximum magnitude of acceleration
                f.append(max([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))                      #Maximum magnitude of gyroscope

                f.append(min([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,1]**2,splice[:,2]**2,splice[:,3]**2)]))                      #Minmum magnitude of acceleration
                f.append(min([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))                      #Minimum magnitude of gyroscope
                

                #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]))))   #Average Fourier transform of acceleration magnitude
                #f.append(np.mean(np.array(fft([math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,7]**2,splice[:,8]**2,splice[:,9]**2)]))))   #Average Fourier transform of gyroscope magnitude
                
                feats.append(f)
                num_sample+=1
        
        l = (num_sample)*[l]
        labels += l
    print(len(feats))
    with open(outdir + "features.csv", 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(feats)):
            writer.writerow(feats[i] + [labels[i]])

    return (feats, labels)



if __name__ == "__main__":


    data = load_all('experiment/')

    #Seizure vs Technology
    (feat, lab) = feature_extraction([data[0], data[1]], [1,0], 'experiment/features/')
    separate(feat, lab)

    #Seizure vs Eating
    (feat, lab) = feature_extraction([data[0], data[2]], [1,0], 'experiment/features/')
    separate(feat, lab)

    #Seizure vs Coversation
    (feat, lab) = feature_extraction([data[0], data[3]], [1,0], 'experiment/features/')
    separate(feat, lab)

    #Seizure vs Non-Seizure
    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/')
    separate(feat, lab)
    
    
    #Vary Threshold

    """
    best = [0,0]
    aucs = []
    f1s = []
    lenfeats = []
    biggest = 0
    for i in range(1,101):
        print(i)
        (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/', i/100.0)
        (auc,f1) = separate(feat, lab)
        if auc > best[0]:
            best = [auc, i]
        aucs.append(auc)
        f1s.append(f1)
        lenfeats.append(len(feat))
        if i == 100: biggest = len(feat)
    

    plt.figure()
    plt.plot(range(1,101), aucs, label="AUC Scores")
    plt.plot(range(1,101), f1s, label="F1 Scores")
    plt.plot(range(1,101), [x / float(biggest) for x in lenfeats], label="Percentage of Samples")
    plt.legend(loc="lower right")
    print("Best threshold is " + str(best[0]) + " at " + str(best[1]))
    plt.show()
    """