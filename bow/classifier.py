import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, io


clf, classes_names, stdSlr, k, voc = joblib.load("bofnew.pkl")
image_paths = []
image_classes = []

#train_path = "dataset/test/"
train_path = "dataset/train/"
training_names = os.listdir(train_path)
class_id=0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    
des_list = []

for image_path in image_paths:
    #print image_paths
    im = io.imread(image_path)
    image = color.rgb2gray(im)
    image = cv2.resize(image, (250, 250)) 
    #print image_path
    fd, des = hog(image, orientations=8, pixels_per_cell=(16,16),
                    cells_per_block=(1, 1), visualise=True)
    #print fd.shape
    fd = np.float32(fd)
    fd=fd.reshape(72,25)
    des_list.append((image_path, fd)) 
    
# Descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor)) 

#print "Here"
#print voc.shape
test_features = np.zeros((len(image_paths), k), "float32")
test_labels=[]
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    name = image_paths[i].split("/")[2]
    test_labels.append(name)
    for w in words:
        test_features[i][w] += 1
    

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')


test_features = stdSlr.transform(test_features)


predictions =  [classes_names[i] for i in clf.predict(test_features)]

cont_error=0
for ind in range(len(predictions)):
    if predictions[ind]!=test_labels[ind]:
        print (predictions[ind],test_labels[ind])
        cont_error+=1

