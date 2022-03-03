import argparse
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
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from process_raw import extract_ncent, process_all
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



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
    print(datadir)
    for file in os.listdir(datadir):
        filename = os.fsdecode(file)
        
        if filename.endswith(".csv") and "data" in filename and "datafill" not in filename:   #Error handling to dodge hidden files and subdirectories
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







def separate_om(features, labels, outdir=None, title=None, n=None):
    """
    Run an LDA on the provided features and labels
    Use another model besides LDA
    """
    train_feat, test_feat, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
    
    clf = LogisticRegression(random_state=42)

    
    clf.fit(train_feat, train_labels)
        
    preds = clf.predict(test_feat)

    
    probs = clf.predict_proba(test_feat)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    auc = metrics.roc_auc_score(test_labels, probs)

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')
    #print(scores)
    f1 = np.average(scores)
    print('AUC: %.3f' % auc)
    print('F1: %.3f' % f1)

    metrics.plot_roc_curve(clf, test_feat, test_labels)


    if n== 0: lab = "Technology"
    elif n==1: lab = "Eating"
    elif n==2: lab = "Conversation"
    else: lab = "Non-Seizure"
    if title != None: plt.title(title)
    if outdir != None: plt.savefig(outdir + lab + "/" + lab + "ROC", dpi=600)

    plt.figure()
    table = plt.table(cellText = [[f1, len(features)]], colLabels=("F1", "#Features"), loc="center")
    plt.axis('off')
    plt.grid('off')
    if outdir != None: plt.savefig(outdir + lab + "/" + lab + "LDAScores", dpi=600, bbox_inches="tight")
    
    return f1

def separate(features, labels, outdir=None, title=None, n=None):
    """
    Run an LDA on the provided features and labels
    """
    #Split into train and test featuers
    train_feat, test_feat, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
    
    #Define the classifier
    clf = LinearDiscriminantAnalysis()
    clf.fit_transform(train_feat, train_labels)
    clf.transform(test_feat)
    
    
    #Generate AUC score
    preds = clf.predict(test_feat)
    probs = clf.predict_proba(test_feat)
    probs = probs[:, 1]
    auc = metrics.roc_auc_score(test_labels, probs)

    #Use 5-Fold CV to generate F1 Macro score
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')
    f1 = np.average(scores)
   
    print('AUC: %.3f' % auc)
    print('F1: %.3f' % f1)

    #Plot ROC Curve
    metrics.plot_roc_curve(clf, test_feat, test_labels)


    if n== 0: lab = "Technology"
    elif n==1: lab = "Eating"
    elif n==2: lab = "Conversation"
    else: lab = "Non-Seizure"

    #Save ROC Curve
    if title != None: plt.title(title)
    if outdir != None: plt.savefig(outdir + lab + "/" + lab + "ROC", dpi=600)

    #Save F1 and AUC scores
    plt.figure()
    table = plt.table(cellText = [[f1, auc, len(features)]], colLabels=("F1", "AUC", "#Features"), loc="center")
    plt.axis('off')
    plt.grid('off')
    if outdir != None: plt.savefig(outdir + lab + "/" + lab + "LDAScores", dpi=600, bbox_inches="tight")

    
    importance = 10*clf.coef_[0]

    #Plot histogram
    if outdir != None: plot_histo(features, labels, outdir, importance, lab, title=title)
    
    return (auc,f1)


def plot_histo(features, labels, outdir, coef, lab, title=None):
    """
    Plot histogram and corresponding feature bar chart
    """
    mapped = []
    
    #Normalize coefficients
    norm_coef = coef / np.linalg.norm(np.array(coef))
    #Save top features to a table
    feature_table(norm_coef, outdir, lab)
    
    #Project onto axis of maximal separation by taking the dot product of the feature coefficient and feature values
    features = np.array(features).dot(norm_coef.T)

    feat_1 = []
    feat_0 = []

    #Separate into 1 labeled and 0 labeled features
    for (feat, l) in zip(features,labels):
        if l==0: feat_1.append(feat)
        else: feat_0.append(feat)
            


    plt.figure()
    if title != None: plt.title(title)
    #Plot overlapping histograms
    feat_1 = np.array(feat_1)
    feat_0 = np.array(feat_0)
    plt.hist(feat_1, bins=50, color='red', stacked=True, alpha=0.8, ec='black', density=False, label=lab, weights=np.zeros_like(feat_1) + 1. / feat_1.size)
    plt.hist(feat_0, bins=50, color='blue', stacked=True, alpha=0.8, ec='black', density=False, label="Seizure", weights=np.zeros_like(feat_0) + 1. / feat_0.size)
    plt.legend(loc="upper left")
    plt.xlabel("Separation")
    plt.ylabel("Frequency")
    xmin, xmax = plt.xlim()

    plt.savefig(outdir + lab + "/" + lab + "Histo", dpi=600)





def feature_table(coef, outdir, lab):
    """
    Save out table of features and coefficients ranked
    """
    eye = ["Average Eye Classification", "# Unique Eye Classificaitons", "# Classifcation Changes", "#Low-Confidence Samples","% Top/Bottom Row",  "% Left/Right Column"] 

    head = ["Average Accelerometer Magnitude", "Average Gyroscope Magnitude", "Std Accelerometer Magnitude"
    , "Std Gyroscope Magnitude", "Maximum Accelerometer", "Maximum Gyroscope", "Minimum Accelerometer", "Minimium Gyroscope", "Range Accelerometer",
    "Range Gyroscope"]

    features = eye+head

    plt.figure()
    plt.bar(features, coef)
    plt.xticks(rotation=90, fontsize=8)
    plt.savefig(outdir + lab + "/" + lab + "Features", dpi=600)

    #Sort in descending order by absolute value
    sorts = reversed(sorted(zip(coef, features), key=lambda x: abs(x[0])))

    plt.figure()
    sorts = [(sub[1], sub[0]) for sub in sorts]
    table = plt.table(cellText = sorts, colLabels=("Feature", "Normalized Coefficient"), loc="center")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(outdir + lab + "/" + lab + "FeaturesTable", dpi=600, bbox_inches="tight")






def feature_extraction(data, labs, outdir, grid=3, et=.05):
    """
    Extract features from raw data

    data: List of lists, each sublist is a run
    labs: List of len(data), a label for each run

    Returns: List of lists, of relevant features 
    """
    #Set constants
    epoch_size = 130
    feats = []
    labels = []
    epoch_thresh = et
    lc = grid*grid

    for (run, l) in zip(data, labs):
        #Determine number of epochs
        n = math.floor(len(run)/(epoch_size/2))

        num_sample = 0
        #Loop across all epochs
        for i in range(1,n):
            f = []
            #Splice all data into current epochs
            splice = np.array(run[(int)((i-1)*(epoch_size/2)):(int)((i*epoch_size/2)+epoch_size/2)])

            classified = splice[:,0]
            #Check if low-confidence threshold is met
            if (classified[classified==lc].shape[0] / len(classified)) <= epoch_thresh:
                
                #Eye Tracking
                f.append(statistics.mean(classified))                                                                                   #Average class
                f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                f.append((np.diff(classified)!=0).sum())                                                                                #Number of times class changes
                f.append((classified==lc).sum())                                                                                        #Number of low confidence values
                
                if grid > 2:                                                                                        
                    
                    t = 0
                    for i in range(0,grid):
                        t += classified[classified==i].shape[0] + classified[classified==grid*grid-grid+i].shape[0]     # %Left/Right Column
                    f.append(t/len(classified))
                    t=0
                    for i in range(0,grid):
                        t += classified[classified==i*(grid)].shape[0] + classified[classified==i*(grid)+(grid-1)].shape[0] # %Top/Bottom Row
                    f.append(t/len(classified))
                
                
                #f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                #f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right
                
                
                

                #Head Tracking
                
                accel_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,1]**2,splice[:,2]**2,splice[:,3]**2)]
                gyro_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,4]**2,splice[:,5]**2,splice[:,6]**2)]

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
                
                f.append(accel_max - accel_min)         #Range acceleration
                f.append(gyro_max - gyro_min)           #Range gyroscope
                  
                feats.append(f)
                num_sample+=1
        
        l = (num_sample)*[l]
        labels += l
    
    print(len(feats))

    return (feats, labels)



def gen_figs(data,outdir,grid=3):
    """
    Generate ROC Curve, Histogram, LDA Score Table, Feature Table, and Feature Bar Chart for Seizure vs Technology, Conversation, Eating, Non-Seizure
    """
    #Seizure vs Technology
    (feat, lab) = feature_extraction([data[0], data[1]], [1,0], 'experiment/features/',grid)
    l = np.array(lab)
    print(l[l==0].shape[0])
    try: separate(feat, lab, outdir, title="Seizure vs Technology", n=0)
    except Exception as e: print("Failed Seizure vs Technology!")

    #Seizure vs Eating
    (feat, lab) = feature_extraction([data[0], data[2]], [1,0], 'experiment/features/',grid)
    l = np.array(lab)
    print(l[l==0].shape[0])
    try: separate(feat, lab, outdir, title="Seizure vs Eating", n=1)
    except Exception as e: print("Failed Seizure vs Eating!")
    
    #Seizure vs Coversation
    (feat, lab) = feature_extraction([data[0], data[3]], [1,0], 'experiment/features/',grid)
    l = np.array(lab)
    print(l[l==0].shape[0])
    try: separate(feat, lab, outdir, title="Seizure vs Conversation", n=2)
    except Exception as e: print("Failed Seizure vs Conversation!")

    #Seizure vs Non-Seizure
    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/',grid)
    l = np.array(lab)
    print(l[l==1].shape[0])
    return separate(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)


def gen_figs_us(data,outdir,grid=3,N=None):   
    """
    Generate figures for all analyses using naive undersampling
    """
    #Seizure vs Non-Seizure

    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/',grid)
    nonseiz_f = []
    seiz_f = []
    for (f,l) in zip(feat,lab):
        if l==0: 
            nonseiz_f.append(f)
        else:
            seiz_f.append(f)
    random.shuffle(nonseiz_f)

    feat = nonseiz_f[0:N*len(seiz_f)] + seiz_f
    lab = [0]*N*len(seiz_f)+[1]*len(seiz_f)

    return separate(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)

def gen_figs_os(data,outdir,grid=3,N=None):   
    """
    Generate figures for all analyses using SMOTE oversampling
    """
    #Seizure vs Non-Seizure

    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/',grid)
    
    feat,lab = SMOTE(sampling_strategy=N).fit_resample(feat,lab)

    return separate(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)
    #except Exception as e: print("Failed Seizure vs Non-Seizure!")

def gen_figs_om(data,outdir,grid=3):   
    """
    Generate features using another machine learning model with 10% SMOTE oversampling
    """
    #Seizure vs Non-Seizure
    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/',grid)
    feat,lab = SMOTE(sampling_strategy=.1).fit_resample(feat,lab)

    return separate_om(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)


def threshold_scan(data, outdir):
    """
    Try epoch thresholds 1-100 and output threshold figure
    """

    best = [0,0]
    aucs = []
    f1s = []
    lenfeats = []
    prev_lf = 0
    prev_score = (0,0)
    biggest = 0
    #Scan all thresholds
    for i in range(1,101):
        print(i)
        #Generate AUC and F1 score
        (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/', grid=3,et=i/100.0)
        lenfeats.append(len(feat))

        if len(feat) != prev_lf: (auc,f1) = separate(feat, lab,n=3)
        else: 
            auc = prev_score[0]
            f1 = prev_score[1]
        aucs.append(auc)
        f1s.append(f1)
        prev_score = (auc, f1)
        prev_lf = len(feat)
        
        
        if i == 100: biggest = len(feat)
    
    #Generate figure
    plt.figure()
    figure, axes = plt.subplots(nrows=2, ncols=2)
    figure.suptitle("Threshold Selection")
    ax = plt.subplot(2,1,1)
    plt.plot(range(1,101), aucs, label="AUC Score")
    plt.plot(range(1,101), f1s, label="F1 Score")

    ax = plt.gca()
    ax.set_ylim([.775, 1])
        
    plt.legend(loc="center right")
    ax = plt.subplot(2,1,2)
    ax = plt.gca()
    ax.set_ylim([10, 100])
    plt.plot(range(1,101), [100*(x / float(biggest)) for x in lenfeats], label="Percentage of Epochs Included")
    
    ax.set_xlabel('Low-Confidence Sample Percentage Threshold')
    plt.legend(loc="lower right")
    

    plt.savefig(outdir + "ThresholdScaled", dpi=600, bbox_inches="tight")
    

def gran(experdir):
    """
    Produce F1 and AUC scores for granularity of 3x3 to 15x15
    """
    aucs = []
    f1 = []
    for n in range(3,16):
        extract_ncent(experdir, n)
        data = process_all("experiment/", centroid="centroids_" + str(n) + ".csv", grid=n, limit=5)
        data = load_all('experiment/')
        outdir = "figures/NxN/granular_nohead/"+str(n)+"/"
        (a, f) = gen_figs(data, outdir, n)
        aucs.append(a)
        f1.append(f)
    return(aucs, f1)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", metavar=("Experiment directory"))
    parser.add_argument("-o", metavar=("Output directory"))
    args = parser.parse_args()

    data = load_all(args.e)
    gen_figs(data, args.o)

    
