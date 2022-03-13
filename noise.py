# To run this script, you need to install: 
#       `pip install matplotlib` 
#       `pip install sklearn
#
# Ensure that pip has installed these packages to the PythonPath the IDE is running

from asyncio import wait_for
import csv
import os
import math
import random
import statistics
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredText
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import f1_score
from analysis import class_to_color, plot_histo
from process_raw import classify, full_feature_extraction, load_centroids
from process_raw import load_all as load_all_9

# Adds specified level of noise to data
# noise_scale is the scaling amount of standard deviation to add in noise (e.g. 0.1 -> 10%)
def noise(data, centroids=None, noise_level: float=0.1, plot=True): 
    
    conf_thresh = .7   #confidence threshold
    n = 1
    noisy_data = []
    original_data = []
    percent_correct_list = []
    x_data = []
    y_data = []

    # Find stdev of data
    for original_data_run in data:
        for sample in original_data_run:
            if ((not math.isnan(sample[0])) and (not math.isnan(sample[1]))):
                x_data = np.append(x_data, sample[0])
                y_data = np.append(y_data, sample[1])

    #Find x and y standard deviations
    x_stdev = statistics.stdev(x_data)
    y_stdev = statistics.stdev(y_data)
    average_stdev = statistics.mean([x_stdev, y_stdev])

    for original_data_run in data:

        labelled_original = []
        labelled_noisy = []
        classified_noisy = []
        noise_to_be_averaged = []

        if plot:
            plt.figure(n)
            for sample in original_data_run:
                # Ignore NaN values
                if ((not math.isnan(sample[0])) and (not math.isnan(sample[1]))):
                    # Plot original data
                    plt.plot(sample[0], sample[1], ".", color = "blue")

        # Add noise to original data
        noisy_data_run = copy.deepcopy(original_data_run)

        for i in range(len(original_data_run)):
            x_noise = np.random.normal(0, noise_level*x_stdev)
            y_noise = np.random.normal(0, noise_level*y_stdev) 
            noise_to_be_averaged = np.append(noise_to_be_averaged, (abs(x_noise), abs(y_noise)))
            noisy_x_coord = original_data_run[i][0] + x_noise
            noisy_y_coord = original_data_run[i][1] + y_noise
            
            # Ignore NaN values
            if (not math.isnan(original_data_run[i][0])) and (not math.isnan(original_data_run[i][1])):
                noisy_data_run[i][0] = noisy_x_coord
                noisy_data_run[i][1] = noisy_y_coord
                if plot:
                    # Plot noisy data
                    plt.plot(noisy_x_coord, noisy_y_coord, ".", color = "red", alpha = 0.3)
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title(f"Noise = {noise_level:.2f}σ")

        if plot:
            #Plot centroids
            if centroids != None:
                for (x,y) in centroids:
                    plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y])))
        
        #Classify original data
        classified_original = classify(centroids, original_data_run, conf_thresh)
        #Classify noisy data
        classified_noisy = classify(centroids, noisy_data_run, conf_thresh)

        # Full datasets
        noisy_data.append(classified_noisy)
        original_data.append(classified_original)

        #Extract grid number from original
        for sample in classified_original: 
            labelled_original = np.append(labelled_original, sample[0])
        # Extract grid number from classified data
        for sample in classified_noisy: 
            labelled_noisy = np.append(labelled_noisy, sample[0])

        #Calculate percent correct between labelled original and noisy
        percent_correct = round((sum(1 for a,b in zip(labelled_noisy, labelled_original) if a ==b)/len(labelled_original) * 100),2)
        percent_correct_list = np.append(percent_correct_list, percent_correct)

        #Calculate average noise variance
        average_noise_variance = (statistics.mean(noise_to_be_averaged))**2

        print(f"Run {n}: {percent_correct}% similar with noise variance {average_noise_variance}")

        n += 1

    # Deal with nan values
    for run in noisy_data:
        for sample in run:
            if math.isnan(sample[1]):
                sample[1] = 9
                sample[2] = 9
                sample[3] = 0
    
    if plot:
        plt.show()

    # Average percent correct
    average_percent_correct = np.mean(percent_correct_list)

    return(original_data, noisy_data, average_noise_variance, average_percent_correct, average_stdev)

def plot_data(data, centroids):

    n = 1

    for original_data_run in data:
        plt.figure(n)
        for sample in original_data_run:
            # Ignore NaN values
            if ((not math.isnan(sample[0])) and (not math.isnan(sample[1]))):
            # Plot original data
                plt.plot(sample[0], sample[1], ".", color = "blue")
        for (x,y) in centroids:
                    plt.plot(x,y,'*', color=class_to_color(centroids.index([x,y])))
        n = n+1

    plt.show()

def load_all(experdir, subj_num):
    """
    Loads all the data from the experiment. Specific to the way the output files are saved

    experdir: String, path to experiment directory

    Returns [ [[conversation_0],...,[conversation_n]], [[eating_0],...,[eating_n]], [[technology_0],..,[technology_n]], [[seizure_0],..,[seizure_n]] ]
    """
    data = [ [], [], [], []] 
    for dirName, subdirList, fileList in os.walk(experdir):
        if f"subj{subj_num}" in dirName:
            print(dirName)
            read = load(dirName)
            for i in range(len(read)):
                if i == 0: j = 1
                elif i == 1: j = 3
                elif i == 2: j = 0
                elif i == 3: j = 2
                else: j = 0
                
                data[j].extend(read[i])
               
    return data

def load(datadir):
    """ 
    Loads in data from specified directory
    
    datadir: string, path to csv data


    Returns: List of lists, sublists are data from each .csv file in the specified directory
    """
    data = []

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
            
    return data

def separate(features, labels, outdir, kfoldCV=True, classified_features=True, 
            head_features=True, full_features=False, plot_roc=False, plot_coef=False):
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

    if kfoldCV:
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42) # new
        scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')
        f1 = np.average(scores)
    else:
        f1 = f1_score(test_labels, preds, average='macro')

    print('AUC: %.3f' % auc)
    print('F1: %.3f' % f1)

    if plot_roc:
        metrics.plot_roc_curve(clf, test_feat, test_labels)        
        plt.show()
        
    if plot_coef:
        importance = 10*clf.coef_[0]
        sorts = features_vs_noise(importance, classified_features=classified_features, head_features=head_features, full_features=full_features)
        
    return (auc,f1,sorts)

def feature_extraction(data, labs, outdir, et=.25, classified_features=True, head_features=True, full_features=False):
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
    if classified_features: offset = 1
    else: offset = 0

    if full_features:
        classified_features=True
        head_features=True

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

                #Full Eye Tracking Features
                if full_features:
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

                # Classified eye tracking features
                if classified_features:
                    classified = splice[:,0]
                    f.append(statistics.mean(classified))                                                                                   #Average class
                    f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                    f.append((np.diff(classified)!=0).sum())  
                    f.append((splice[:,0+offset]==9).sum())                                                                                #Number of times class changes             
                    f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                    f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right

                #Head Tracking
                if head_features:
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
                    
                feats.append(f)
                num_sample+=1
        
        l = (num_sample)*[l]
        labels += l
    print(len(feats))
    return(feats,labels)

def separability_vs_noise(subjects, datadir, outdir, centroids_name, min_noise, max_noise, 
                        noise_step, eyeonly=False, kfoldCV=True, classified_features=True,
                        head_features=True, full_features=False, plot_F1vsNoise=True,
                        plot_AUROCvsNoise=True, plot_noise=False, plot_roc=False, plot_coef=False):
    """
    Runs feature extraction and separation for range of noise 
    scaling factors (e.g. 0.1, 0.5, 2) from min_noise to 
    max_noise, incrementing by noise_step

    Parameters:
    Subjects: List of subject number(s) (e.g. 1,2,3,etc.) to run script on
    Datadir: Location of data containing subj folders (e.g. "experiment")
    Centroids_name: Name of centroid file (e.g. "centroids_0.csv")
    min_noise: minimum noise scaling level (e.g. 0.1 -> 10% of std deviation of data is included as noise)
    max_noise: maximum noise scaling level
    noise_step: step size for noise 
    eyeonly: Separate on either eye and head features or eye features only
    kfoldCV: Use 5 fold cross validation (recommended) in separation or not
    plot_F1vsNoise: Plots F1 for varying noise levels
    plot_AUROCvsNoise: Plots AUROC for varying noise levels
    plot_noise: Plot noisy X,Y data 
    plot_roc: Plot ROC curve for single noise level 

    Plots auroc, f1, and percent accuracy vs noise

    Returns F1 score, AUROC score, percent correct, noise variance, average standard deviation and noise level for each noise level
    """

    centroids = []
    noise_list = np.arange(start = min_noise, stop = max_noise+noise_step, step = noise_step)
    main_noisy_data = [ [ [],[],[],[] ] for _ in range(len(noise_list))] # full -> noise level -> behavior -> sample
    percent_correct_list = [ [] for _ in range(len(noise_list))]
    average_percent_correct_list = []
    noise_variance_list = [ [] for _ in range(len(noise_list))]
    average_noise_variance_list = []
    average_stdev_list = []
    full_f1 = []
    full_auroc = []
    first_iter = True

    feature_1_list = []
    feature_2_list = []  
    feature_3_list = []
    feature_4_list = []
    feature_5_list = []
    feature_6_list = []
    feature_7_list = []  
    feature_8_list = []
    feature_9_list = []
    feature_10_list = []
    feature_11_list = []
    feature_12_list = []  
    feature_13_list = []
    feature_14_list = []
    feature_15_list = []
    feature_16_list = []

    # Add noise to each subject
    for subj_num in subjects:
        original_untouched_data = load_all(f"{datadir}/", subj_num) 
        centroids = load_centroids(f"{datadir}/subj{subj_num}/{centroids_name}")
        noise_index = 0
        first_iter = True
        for noise_scale in noise_list:
            print(f"Noise Scale = {noise_scale:.2f}")
            print(f"Noise Index = {noise_index}")
            (original_data2, noisy_data, average_noise_variance, percent_correct, average_stdev) = noise(original_untouched_data, centroids=centroids, noise_level=noise_scale, plot=plot_noise)
            if first_iter:
                average_stdev_list = np.append(average_stdev_list, average_stdev)
                first_iter = False
            percent_correct_list[noise_index] = np.append(percent_correct_list[noise_index], percent_correct)
            noise_variance_list[noise_index] = np.append(noise_variance_list[noise_index], average_noise_variance)
            main_noisy_data[noise_index][0].extend(noisy_data[0])
            main_noisy_data[noise_index][1].extend(noisy_data[1])
            main_noisy_data[noise_index][2].extend(noisy_data[2])
            main_noisy_data[noise_index][3].extend(noisy_data[3])
            noise_index=noise_index+1

    

    first_iter = True
    # Find F1 and AUROC for full dataset
    for i in range(len(noise_list)):
        # if eyeonly:
        #     (feat,lab) = eye_tracking_feature_extraction(main_noisy_data[i], [1,0,0,0], f"{datadir}/features1/", c = True)
        # else:
        #     (feat,lab) = full_feature_extraction(main_noisy_data[i], [1,0,0,0], f"{datadir}/features1/", c = True)
        (feat,lab) = feature_extraction(main_noisy_data[i], [1,0,0,0], f"{datadir}/features1/", classified_features=classified_features, head_features=head_features, full_features=full_features)
        (auroc, f1, sorts) = separate(feat, lab, outdir, kfoldCV=kfoldCV, plot_roc=plot_roc, plot_coef=plot_coef, classified_features=classified_features, head_features=head_features, full_features=full_features)
        full_f1.append(f1)
        full_auroc.append(auroc)
        average_percent_correct_list = np.append(average_percent_correct_list, statistics.mean(percent_correct_list[i]))
        average_noise_variance_list = np.append(average_noise_variance_list, statistics.mean(noise_variance_list[i]))

        # Grab 5 most important features to track
        if first_iter:
            feature_1 = sorts[0][0]
            feature_2 = sorts[1][0]
            feature_3 = sorts[2][0]
            feature_4 = sorts[3][0]
            feature_5 = sorts[4][0]
            feature_6 = sorts[5][0]
            feature_7 = sorts[6][0]
            feature_8 = sorts[7][0]
            feature_9 = sorts[8][0]
            feature_10 = sorts[9][0]
            feature_11 = sorts[10][0]
            feature_12 = sorts[11][0]
            feature_13 = sorts[12][0]
            feature_14 = sorts[13][0]
            feature_15 = sorts[14][0]
            feature_16 = sorts[15][0]

            first_iter = False

        for feature in sorts:
            if feature[0] == feature_1:
                feature_1_list.append(abs(feature[1]))
            elif feature[0] == feature_2:
                feature_2_list.append(abs(feature[1]))
            elif feature[0] == feature_3:
                feature_3_list.append(abs(feature[1]))
            elif feature[0] == feature_4:
                feature_4_list.append(abs(feature[1]))
            elif feature[0] == feature_5:
                feature_5_list.append(abs(feature[1]))
            elif feature[0] == feature_6:
                feature_6_list.append(abs(feature[1]))
            elif feature[0] == feature_7:
                feature_7_list.append(abs(feature[1]))
            elif feature[0] == feature_8:
                feature_8_list.append(abs(feature[1]))
            elif feature[0] == feature_9:
                feature_9_list.append(abs(feature[1]))
            elif feature[0] == feature_10:
                feature_10_list.append(abs(feature[1]))
            elif feature[0] == feature_11:
                feature_11_list.append(abs(feature[1]))
            elif feature[0] == feature_12:
                feature_12_list.append(abs(feature[1]))
            elif feature[0] == feature_13:
                feature_13_list.append(abs(feature[1]))
            elif feature[0] == feature_14:
                feature_14_list.append(abs(feature[1]))
            elif feature[0] == feature_15:
                feature_15_list.append(abs(feature[1]))
            elif feature[0] == feature_16:
                feature_16_list.append(abs(feature[1]))
        
    plt.figure()
    ax=plt.subplot()
    plt.plot(noise_list, feature_1_list, color='blue', marker='o')
    plt.plot(noise_list, feature_2_list, color='red', marker='^')
    plt.plot(noise_list, feature_3_list, color='green', marker='s')
    plt.plot(noise_list, feature_4_list, color='purple', marker='P')
    plt.plot(noise_list, feature_5_list, color='black', marker='*')

    plt.plot(noise_list, feature_6_list, color='cyan', marker='v')
    # plt.plot(noise_list, feature_7_list, color='magenta', marker='>')
    # plt.plot(noise_list, feature_8_list, color='dodgerblue', marker='1')
    # plt.plot(noise_list, feature_9_list, color='seagreen', marker='2')
    # plt.plot(noise_list, feature_10_list, color='lime', marker='3')
    # plt.plot(noise_list, feature_11_list, color='sienna', marker='4')
    # plt.plot(noise_list, feature_12_list, color='khaki', marker='D')
    # plt.plot(noise_list, feature_13_list, color='gold', marker='X')
    # plt.plot(noise_list, feature_14_list, color='navy', marker='+')
    # plt.plot(noise_list, feature_15_list, color='fuchsia', marker='x')
    # plt.plot(noise_list, feature_16_list, color='crimson', marker='d')

    plt.xlabel("Noise")
    plt.ylabel("Feature Coefficient")

    blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=10, label=feature_1)
    red_line = mlines.Line2D([], [], color='red', marker='^', markersize=10, label=feature_2)
    green_line = mlines.Line2D([], [], color='green', marker='s', markersize=10, label=feature_3)
    purple_line = mlines.Line2D([], [], color='purple', marker='P', markersize=10, label=feature_4)
    black_line = mlines.Line2D([], [], color='black', marker='*', markersize=10, label=feature_5)
    cyan_line = mlines.Line2D([], [], color='cyan', marker='v', markersize=10, label=feature_6)
    # magenta_line = mlines.Line2D([], [], color='magenta', marker='>', markersize=10, label=feature_7)
    # dodgerblue_line = mlines.Line2D([], [], color='dodgerblue', marker='1', markersize=10, label=feature_8)
    # seagreen_line = mlines.Line2D([], [], color='seagreen', marker='2', markersize=10, label=feature_9)
    # lime_line = mlines.Line2D([], [], color='lime', marker='3', markersize=10, label=feature_10)
    # sienna_line = mlines.Line2D([], [], color='sienna', marker='4', markersize=10, label=feature_11)
    # khaki_line = mlines.Line2D([], [], color='khaki', marker='D', markersize=10, label=feature_12)
    # gold_line = mlines.Line2D([], [], color='gold', marker='X', markersize=10, label=feature_13)
    # navy_line = mlines.Line2D([], [], color='navy', marker='+', markersize=10, label=feature_14)
    # fuchsia_line = mlines.Line2D([], [], color='fuchsia', marker='x', markersize=10, label=feature_15)
    # crimson_line = mlines.Line2D([], [], color='crimson', marker='d', markersize=10, label=feature_16)

    #, magenta_line, dodgerblue_line, seagreen_line, lime_line,  sienna_line, khaki_line, gold_line, navy_line, fuchsia_line, crimson_line
    ax.legend(handles=[blue_line, red_line, green_line, purple_line, black_line, cyan_line], loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3)
    #plt.tight_layout()

    plt.show()

    average_average_stdev = statistics.mean(average_stdev_list)

    # Title details for plotting
    if eyeonly:
        title_info = "Eye Features Only"
    else:
        title_info = "Eye & Head Features"
    
    if kfoldCV:
        title_info = title_info + ", 5-Fold CV"
    else:
        title_info = title_info + ", No 5-Fold CV"

    # Plot F1 vs Noise
    if plot_F1vsNoise:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(noise_list, full_f1, color='blue', marker='o')
        ax2.plot(noise_list, average_percent_correct_list, color='red', marker='o')

        ax1.set_title(f"F1 vs Noise, {title_info}")
        ax1.set_xlabel("Noise (amount of standard deviation)")
        ax1.set_ylabel("F1", color='blue')
        ax2.set_ylabel("Percent Accuracy", color='red')

        at = AnchoredText(
        f"σ={average_average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

    # Plot AUROC vs Noise
    if plot_AUROCvsNoise:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(noise_list, full_auroc, color='blue', marker='o')
        ax2.plot(noise_list, average_percent_correct_list, color='red', marker='o')

        ax1.set_title(f"AUROC vs Noise, {title_info}")
        ax1.set_xlabel("Noise (amount of standard deviation)")
        ax1.set_ylabel("AUROC", color='blue')
        ax2.set_ylabel("Percent Accuracy", color='red')

        at = AnchoredText(
        f"σ={average_average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

    print(f"F1 Scores: {full_f1}")
    print(f"AUROC Scores: {full_auroc}")
    print(f"Percent Correct Scores: {average_percent_correct_list}")
    print(f"Noise Variances: {average_noise_variance_list}")

    plt.show()

    return full_f1, full_auroc, average_percent_correct_list, average_noise_variance_list, average_average_stdev, noise_list

def features_vs_noise(coef, classified_features=True, head_features=True, full_features=False):

    full = ["Average X", "Average Y", "Max X", "Min X", "Max Y", "Min Y", "Range X", "Range Y", "Max Distance btwn Points", "Min Distance btwn Points", 
    "Range Distance btwn Points", "Average Distance btwn all Points", "Average Distance btwn Adjacent Points", "Average Conf", "Std Conf", "# Low-Confidence Samples"]
    
    eye = ["Average Eye Classification", "# Unique Eye Classificaitons", "# Classifcation Changes", "# Low-Confidence Samples", 
    "% Top/Bottom Row",  "% Left/Right Column"] 

    head = ["Average Accelerometer Magnitude", "Average Gyroscope Magnitude", "Std Accelerometer Magnitude"
    , "Std Gyroscope Magnitude", "Maximum Accelerometer", "Maximum Gyroscope", "Minimum Accelerometer", "Minimium Gyroscope", "Range Accelerometer",
    "Range Gyroscope"]

    if full_features:
        features = eye + head + full
    elif classified_features and head_features:
        features = eye + head
    elif classified_features:
        features = eye
    elif head_features:
        features = head

    norm_coef = coef / np.linalg.norm(np.array(coef))
    sorts = reversed(sorted(zip(norm_coef, features), key=lambda x: abs(x[0])))
    sorts = [(sub[1], sub[0]) for sub in sorts]

    print(sorts)

    return sorts

if __name__ == "__main__":

    subjects = [1,2,3,4,5,6,7,8]
    datadir = "noisy_experiment"
    outdir = "noisy_experiment/"
    centroids_name = "centroids_0.csv"
    min_noise = 0.8
    max_noise = 0.8
    noise_step = 0.2

    # Find F1 and AUROc for range of noise levels
    separability_vs_noise(subjects, datadir, outdir, centroids_name, min_noise, max_noise, noise_step, 
                            eyeonly=False, kfoldCV=True, plot_F1vsNoise=True,
                            plot_AUROCvsNoise=True, plot_noise=True,  plot_roc=False, plot_coef=True)

    # Comparing different analysis combinations (e.g. eye & head features vs eye features only, 5foldCV vs no 5foldCV)
    # better_full_f1, better_full_auroc, better_average_percent_correct_list, better_average_noise_variance_list, better_average_average_stdev, better_noise_list = separability_vs_noise(subjects, 
    #                         datadir, centroids_name, min_noise, max_noise, noise_step, eyeonly=False, 
    #                         kfoldCV=True, plot_F1vsNoise=False, plot_AUROCvsNoise=False) 
    # worse_full_f1, worse_full_auroc, worse_average_percent_correct_list, worse_average_noise_variance_list, worse_average_average_stdev, worse_noise_list = separability_vs_noise(subjects, 
    #                         datadir, centroids_name, min_noise, max_noise, noise_step, eyeonly=True, 
    #                         kfoldCV=True, plot_F1vsNoise=False, plot_AUROCvsNoise=False) 

    # # Plot F1 vs Noise
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    # ax1.plot(better_noise_list, better_full_f1, color='blue', marker='s')
    # ax1.plot(worse_noise_list, worse_full_f1, color = 'green', marker='^')
    # ax2.plot(better_noise_list, better_average_percent_correct_list, color='red', marker='o')

    # ax1.set_title(f"F1 vs Noise")
    # ax1.set_xlabel("Noise (amount of standard deviation)")
    # ax1.set_ylabel("F1", color='blue')
    # ax2.set_ylabel("Percent Accuracy", color='red')

    # at = AnchoredText(
    # f"σ={better_average_average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax1.add_artist(at)

    # blue_line = mlines.Line2D([], [], color='blue', marker='s', markersize=15, label='Eye & Head, 5FoldCV')
    # green_line = mlines.Line2D([], [], color='green', marker='^', markersize=15, label='Eye Only, 5FoldCV')

    # ax1.legend(handles=[blue_line, green_line], loc='upper right')

    # # Plot AUROC vs Noise
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    
    # ax1.plot(better_noise_list, better_full_auroc, color='blue', marker='s')
    # ax1.plot(worse_noise_list, worse_full_auroc, color='green', marker='^')
    # ax2.plot(better_noise_list, better_average_percent_correct_list, color='red', marker='o')

    # ax1.set_title(f"AUROC vs Noise")
    # ax1.set_xlabel("Noise (amount of standard deviation)")
    # ax1.set_ylabel("AUROC", color='blue')
    # ax2.set_ylabel("Percent Accuracy", color='red')

    # at = AnchoredText(
    # f"σ={better_average_average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax1.add_artist(at)

    # blue_line = mlines.Line2D([], [], color='blue', marker='s', markersize=15, label='Eye & Head, 5FoldCV')
    # green_line = mlines.Line2D([], [], color='green', marker='^', markersize=15, label='Eye Only, 5FoldCV')

    # ax1.legend(handles=[blue_line, green_line], loc='upper right')

    # plt.show()
    


