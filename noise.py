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
from matplotlib.offsetbox import AnchoredText
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from analysis import class_to_color
from process_raw import classify, full_feature_extraction, load_centroids
from process_raw import load_all as load_all_9

# Adds specified level of noise to data
# noise_scale is the scaling amount of standard deviation to add in noise (e.g. 0.1 -> 10%)
def noise(data, centroids=None, noise_level: float=0.1, plot=True): # used to be noise_level: float=0.1
    
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

        #Add noise to original data
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
                    plt.plot(noisy_x_coord, noisy_y_coord, ".", color = "red", alpha = 0.3)

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

def load_all(experdir):
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

def separate(features, labels, plot=False):
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
    print('F1: %.3f' % f1)

    if plot:
        metrics.plot_roc_curve(clf, test_feat, test_labels)
            

            #plt.figure()
            #importance = 10*clf.coef_[0]
            #print(importance)
            # for i,v in enumerate(importance):
            #     print('Feature: %0d, Score: %.5f' % (i,v))
            #full = ['Average Eye', '#Unique', '#Changes', '#Low-Conf', '%TB', '%LR' 'Avg Accel Mag', 'Avg Gyro Mag', 'Std Accel Mag', 'Std Gyro Mag', 'Accel Max', 'Gyro Max', 'Accel Min', 'Gyro Min']
            #justeye = ['Average Eye Class', '#Unique', '#Class Changes', '#Low-Confidence']
            #plt.bar(full, importance)
            #plt.xticks(rotation=90, fontsize=8)
        
        plt.show()

    #plot_histo(features, labels, importance)
    #auc = 0
    return (auc,f1)

def eye_tracking_feature_extraction(data, labs, outdir, et=.15, c=False):
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

                f.append((splice[:,0+offset]==9).sum())   
                f.append(statistics.mean(conf))

                if c:
                    classified = splice[:,0]
                    f.append(statistics.mean(classified))                                                                                   #Average class
                    f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                    f.append((np.diff(classified)!=0).sum())                                                                                #Number of times class changes             
                    f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                    f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right

                #Head Tracking
                # accel_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,3+offset]**2,splice[:,4+offset]**2,splice[:,5+offset]**2)]
                # gyro_mag = [math.sqrt(x+y+z) for (x,y,z) in zip(splice[:,6+offset]**2,splice[:,7+offset]**2,splice[:,8+offset]**2)]

                # f.append(statistics.mean(accel_mag))
                # f.append(statistics.mean(gyro_mag))  
                
                # f.append(statistics.stdev(accel_mag))         #Standard Deviation of acceleration
                # f.append(statistics.stdev(gyro_mag))         #Standard Deviation magnitude of gyroscope
                
                # accel_max = max(accel_mag)
                # gyro_max = max(gyro_mag)

                # accel_min = min(accel_mag)
                # gyro_min = min(gyro_mag)

                # f.append(accel_max)                      #Maximum magnitude of acceleration
                # f.append(gyro_max)                      #Maximum magnitude of gyroscope

                # f.append(accel_min)                      #Minmum magnitude of acceleration
                # f.append(gyro_min)                      #Minimum magnitude of gyroscope
                
                # f.append(accel_max - accel_min)
                # f.append(gyro_max - gyro_min)      


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

def separability_vs_noise(original_data, centroids, min_noise, max_noise, noise_increment):
    """
    Runs feature extraction and separation for range of noise 
    scaling factors (e.g. 0.1, 0.5, 2) from min_noise to 
    max_noise, incrementing by noise_increment

    Parameters:
    cleaned_original_data: list of lists where each sublist is one behavior. Nan has been replaced with 9's
    original_data: list of lists where each sublist is one behavior. Nan has been left in
    centroids: list of centroids
    min_noise: minimum noise scaling level (e.g. 0.1 -> 10% of std deviation of data is included as noise)
    max_noise: maximum noise scaling level
    noise_increment: step size for noise 

    Plots auroc, f1, and percent accuracy vs noise
    """
    noise_list = np.arange(start = min_noise, stop = max_noise+noise_increment, step = noise_increment)
    noise_list = noise_list.astype(float)
    f1_list = []
    auroc_list = []
    percent_correct_list = []
    average_noise_variance_list = []

    for noise_scale in noise_list:
        (original_data2, noisy_data, average_noise_variance, percent_correct, average_stdev) = noise(original_data, centroids=centroids, noise_level=noise_scale, plot=False)

        (feat,lab) = eye_tracking_feature_extraction(noisy_data, [1,0,0,0], "single_noisy_experiment/features1/", c = True)
        (auroc, f1) = separate(feat, lab)
        print(f"current f1: {f1}")

        auroc_list.append(auroc)
        f1_list.append(f1)
        average_noise_variance_list.append(average_noise_variance)
        percent_correct_list.append(percent_correct)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(noise_list, f1_list, color='blue', marker='o')
    ax2.plot(noise_list, percent_correct_list, color='red', marker='o')

    ax1.set_title("Noise vs F1, Eye Data Only")
    ax1.set_xlabel("Noise (amount of standard deviation)")
    ax1.set_ylabel("F1", color='blue')
    ax2.set_ylabel("Percent Accuracy", color='red')

    at = AnchoredText(
    f"σ={average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(noise_list, auroc_list, color='blue', marker='o')
    ax2.plot(noise_list, percent_correct_list, color='red', marker='o')

    ax1.set_title("Noise vs AUROC, Eye Data Only")
    ax1.set_xlabel("Noise (amount of standard deviation)")
    ax1.set_ylabel("AUROC", color='blue')
    ax2.set_ylabel("Percent Accuracy", color='red')

    at = AnchoredText(
    f"σ={average_stdev:.3f}", prop=dict(size=15), frameon=True, loc='lower left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    print(f"auroc: {auroc_list}")
    print(f"f1: {f1_list}")
    print(f"percent correct: {percent_correct_list}")
    print(f"noise: {noise_list}")
    print(f"average noise variance: {average_noise_variance_list}")

    plt.show()

    
if __name__ == "__main__":
    
    # FULL EXPERIMENT DATA
    # data = load_all_9("full_noisy_experiment/", c=True,cent='/centroids_0.csv')

    # (feat, lab) = full_feature_extraction(data, [1,0,0,0], 'full_noisy_experiment/features/',c=True)
    # separate(feat, lab)

    # SINGLE SUBJECT DATA CLEAN
    # data = load_all_9("single_noisy_experiment/", c=True, cent='/centroids_0.csv')

    # (feat, lab) = full_feature_extraction(data, [1,0,0,0], 'single_noisy_experiment/features/')
    # print(feat)
    # separate(feat, lab)

    # SINGLE SUBJECT DATA CLEAN AND NOISY, SEPARATE ON EYE AND HEAD DATA
    # cleaned_original_data = load_all_9("single_noisy_experiment/", c = True, cent='/centroids_0.csv')
    # original_untouched_data = load_all("single_noisy_experiment/")
    # centroids = load_centroids("single_noisy_experiment/subj1/centroids_0.csv")
    # (original_data2, noisy_data) = noise(original_untouched_data, centroids, noise_level = 0.1, plot=False)

    # (feat1,lab1) = full_feature_extraction(cleaned_original_data, [1,0,0,0], "single_noisy_experiment/features1/", c=True)
    # separate(feat1,lab1)

    # (feat2,lab2) = full_feature_extraction(noisy_data, [1,0,0,0], "single_noisy_experiment/features2/", c=True)
    # separate(feat2,lab2)

    #SINGLE SUBJECT DATA CLEAN AND NOISY, SEPARATE ON ONLY EYE DATA
    # cleaned_original_data = load_all_9("single_noisy_experiment/", c=True, cent='/centroids_0.csv')
    # original_untouched_data = load_all("single_noisy_experiment/")
    # centroids = load_centroids("single_noisy_experiment/subj8/centroids_0.csv")
    # (original_data2, noisy_data) = noise(original_untouched_data, centroids, noise_level=1, plot=False)

    # # for run1, run2 in zip(cleaned_original_data, noisy_data):
    # #     for sample1, sample2 in zip(run1, run2):
    # #         print(f"original: {sample1}, noisy: {sample2}")

    # (feat1,lab1) = eye_tracking_feature_extraction(cleaned_original_data, [1,0,0,0], "single_noisy_experiment/features1/", c=True)
    # separate(feat1,lab1)

    # (feat2,lab2) = eye_tracking_feature_extraction(noisy_data, [1,0,0,0], "single_noisy_experiment/features2/", c=True)
    # separate(feat2,lab2)

    #SINGLE SUBJECT DATA VARYING NOISY
    cleaned_original_data = load_all_9("single_noisy_experiment/", c = True, cent='/centroids_0.csv')
    original_untouched_data = load_all("single_noisy_experiment/")
    centroids = load_centroids("single_noisy_experiment/subj3/centroids_0.csv")

    separability_vs_noise(original_untouched_data, centroids, 0, 5, 0.2)



