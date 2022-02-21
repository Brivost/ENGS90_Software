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
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
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
    print(datadir)
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
        
        plt.savefig('figures/' + str(n) + ".png", dpi=600)
        n+=1
        plt.figure()






def separate(features, labels, outdir, title=None, n=None):
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

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')
    f1 = np.average(scores)
    print('AUC: %.3f' % auc)
    print('F1: %.3f' % f1)

    metrics.plot_roc_curve(clf, test_feat, test_labels)


    if n== 0: lab = "Technology"
    elif n==1: lab = "Eating"
    elif n==2: lab = "Conversation"
    else: lab = "Non-Seizure"
    if title != None: plt.title(title)
    plt.savefig(outdir + lab + "/" + lab + "ROC", dpi=600)

    plt.figure()
    table = plt.table(cellText = [[f1, auc, len(features)]], colLabels=("F1", "AUC", "#Features"), loc="center")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(outdir + lab + "/" + lab + "LDAScores", dpi=600, bbox_inches="tight")

    
    importance = 10*clf.coef_[0]

    plot_histo(features, labels, outdir, importance, lab, title=title)
    
    return (auc,f1)


def plot_histo(features, labels, outdir, coef, lab, title=None):
    mapped = []
    # for feat in features:
    #     for (c, f) in zip(coef, feat):
    #         print(c*f)
    
    norm_coef = coef / np.linalg.norm(np.array(coef))
    feature_table(norm_coef, outdir, lab)
    
    features = np.array(features).dot(norm_coef.T)

    feat_1 = []
    feat_0 = []

    for (feat, l) in zip(features,labels):
        if l==0: feat_1.append(feat)
        else: feat_0.append(feat)
            
    #mapped_1 = [sum([c*f for (c,f) in zip(coef, feat)]) for feat in feat_1]
    #mapped_0 = [sum([c*f for (c,f) in zip(coef, feat)]) for feat in feat_0]

    plt.figure()
    if title != None: plt.title(title)
    plt.hist(feat_1, bins=50, color='red', stacked=True, alpha=0.8, ec='black', density=True, label=lab)
    plt.hist(feat_0, bins=50, color='blue', stacked=True, alpha=0.8, ec='black', density=True, label="Seizure")
    plt.legend(loc="upper left")

    xmin, xmax = plt.xlim()
    mu1, std1 = norm.fit(feat_1)
    mu0, std0 = norm.fit(feat_0)

    x = np.linspace(xmin, xmax, 100)
    p1 = norm.pdf(x, mu1, std1)
    p0 = norm.pdf(x, mu0, std0)

    plt.plot(x, p1, color='red')
    plt.plot(x, p0, color='blue')
    plt.savefig(outdir + lab + "/" + lab + "Histo", dpi=600)





def feature_table(coef, outdir, lab):
    full = ["Average X", "Average Y", "Max X", "Min X", "Max Y", "Min Y", "Range X", "Range Y", "Max Distance btwn Points", "Min Distance btwn Points", 
    "Range Distance btwn Points", "Average Distance btwn all Points", "Average Distance btwn Adjacent Points", "Average Conf", "Std Conf", "# Low-Confidence Samples"]
    eye = ["Average Eye Classification", "# Unique Eye Classificaitons", "# Classifcation Changes", "# Low-Confidence Samples", 
    "% Top/Bottom Row",  "% Left/Right Column"] 

    head = ["Average Accelerometer Magnitude", "Average Gyroscope Magnitude", "Std Accelerometer Magnitude"
    , "Std Gyroscope Magnitude", "Maximum Accelerometer", "Maximum Gyroscope", "Minimum Accelerometer", "Minimium Gyroscope", "Range Accelerometer",
    "Range Gyroscope"]

    ideal = ["% Left/Right Column", "Average Accelerometer Magnitude"]

    features = eye+head



    plt.figure()
    plt.bar(features, coef)
    plt.xticks(rotation=90, fontsize=8)
    plt.savefig(outdir + lab + "/" + lab + "Features", dpi=600)
    #plt.show()
    sorts = reversed(sorted(zip(coef, features), key=lambda x: abs(x[0])))

    plt.figure()
    sorts = [(sub[1], sub[0]) for sub in sorts]
    table = plt.table(cellText = sorts, colLabels=("Feature", "Normalized Coefficient"), loc="center")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(outdir + lab + "/" + lab + "FeaturesTable", dpi=600, bbox_inches="tight")






def feature_extraction(data, labs, outdir, et=.25):
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

    for (run, l) in zip(data, labs):
        n = math.floor(len(run)/(epoch_size/2))

        num_sample = 0
        for i in range(1,n):
            f = []

            splice = np.array(run[(int)((i-1)*(epoch_size/2)):(int)((i*epoch_size/2)+epoch_size/2)])
            classified = splice[:,0]
            if (classified[classified==9].shape[0] / len(classified)) <= epoch_thresh:
                #Eye Tracking
                
                f.append(statistics.mean(classified))   
                #f.append(statistics.stdev(classified))                                                                                #Average class
                f.append(len(np.unique(classified)))                                                                                    #Number of unique classes
                f.append((np.diff(classified)!=0).sum())                                                                                #Number of times class changes
                f.append((classified==9).sum())                                                                                         #Number of poor confidence values
                f.append((classified[classified==0].shape[0] + classified[classified==1].shape[0] + classified[classified==2].shape[0] + classified[classified==6].shape[0] + classified[classified==7].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent top/bottom
                f.append((classified[classified==0].shape[0] + classified[classified==3].shape[0] + classified[classified==6].shape[0] + classified[classified==2].shape[0] + classified[classified==5].shape[0] + classified[classified==8].shape[0]) / len(classified))  #Percent left/right
                

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
                
                f.append(accel_max - accel_min)
                f.append(gyro_max - gyro_min)      
                
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

def gen_figs(data,outdir):
    #Seizure vs Technology
    (feat, lab) = feature_extraction([data[0], data[1]], [1,0], 'experiment/features/')
    try: separate(feat, lab, outdir, title="Seizure vs Technology", n=0)
    except Exception as e: print("Failed Seizure vs Technology!")

    #Seizure vs Eating
    (feat, lab) = feature_extraction([data[0], data[2]], [1,0], 'experiment/features/')
    try: separate(feat, lab, outdir, title="Seizure vs Eating", n=1)
    except Exception as e: print("Failed Seizure vs Eating!")
    #Seizure vs Coversation
    (feat, lab) = feature_extraction([data[0], data[3]], [1,0], 'experiment/features/')
    try: separate(feat, lab, outdir, title="Seizure vs Conversation", n=2)
    except Exception as e: print("Failed Seizure vs Conversation!")

    #Seizure vs Non-Seizure
    (feat, lab) = feature_extraction(data, [1,0,0,0], 'experiment/features/')
    try:separate(feat, lab, outdir, title="Seizure vs Non-Seizure", n=3)
    except Exception as e: print("Failed Seizure vs Non-Seizure!")


if __name__ == "__main__":
    #data = load_all('experiment/')
    outdir = "figures/3x3Analysis/Subs/"

    for j in range(1,9):
        data = [ [], [], [], []] 
        read = load('experiment/subj'+str(j)+"/")
        for i in range(len(read)):
            if i > 0:
                data[i-1].extend(read[i])
            else:
                data[i].extend(read[i])
        path = outdir+"subj"+str(j)+"/"
        if not os.path.isdir(path):
            print(path)
            os.makedirs(path, exist_ok=True)

        if not os.path.isdir(path+"Technology/"): os.makedirs(path + "Technology/", exist_ok=True)
        if not os.path.isdir(path+"Eating/"): os.makedirs(path + "Eating/", exist_ok=True)
        if not os.path.isdir(path+"Conversation/"): os.makedirs(path + "Conversation/", exist_ok=True)
        if not os.path.isdir(path+"Non-Seizure/"): os.makedirs(path + "Non-Seizure/", exist_ok=True)
        gen_figs(data, path)

    
    
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
        if f1 > best[0] and i > 5:
            best = [f1, i]
        aucs.append(auc)
        f1s.append(f1)
        lenfeats.append(len(feat))
        if i == 100: biggest = len(feat)
    

    plt.figure()
    figure, axes = plt.subplots(nrows=2, ncols=2)
    figure.suptitle("Threshold Selection")
    ax = plt.subplot(2,1,1)
    plt.plot(range(1,101), aucs, label="AUC Score")
    plt.plot(range(1,101), f1s, label="F1 Score")
    #ax.title.set_text('Seizure vs Non-Seizure Classifier Performance')
    #ax.set_xlabel('Low-Confidence Sample Percentage Threshold')
    plt.legend(loc="lower left")
    ax = plt.subplot(2,1,2)
    
    plt.plot(range(1,101), [100*(x / float(biggest)) for x in lenfeats], label="Percentage of Epochs Included")
    
    ax.set_xlabel('Low-Confidence Sample Percentage Threshold')
    #ax.set_ylabel('Percentage of Epochs Included')
    plt.legend(loc="lower right")
    
    print("Best threshold is " + str(best[0]) + " at " + str(best[1]))
    plt.show()
    """