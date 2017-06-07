import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, io

train_path = "dataset/train/"
training_names = os.listdir(train_path)
print train_path,training_names


image_paths = []
image_classes = []
class_id = 0 # Id from the class
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

print "Se van a procesar lo siguientes artistas"
#print image_classes,image_paths

des_list = []
contador = 1
# HOG
print "Start Here"
for image_path in image_paths:
    #print contador
    print image_path
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
    contador+=1

print "End Process"

#Store all descriptors as columns
descriptors = des_list[0][1]
#print descriptors
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

#print descriptors.shape
# Computes k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    #print voc.shape
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

clf = linear_model.SGDClassifier()
clf.fit(im_features, np.array(image_classes))

# Save the SVM.SGD
joblib.dump((clf, training_names, stdSlr, k, voc), "bofnew_SGD.pkl", compress=3)    
    
