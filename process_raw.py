
import csv
import math
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

def process(rawdir, outdir, cent, max_feat=False):

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
                #writer.writerow(read)

    return (data, centroids)




if __name__ == "__main__":

    process("test_scripts/", "test_scripts/", "centroids_0.csv")
