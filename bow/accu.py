import os,glob
import pickle
import shutil
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

predictions,test_features,test_labels=joblib.load("pre_test_labels")
print accuracy_score(test_labels, predictions)
predictions,test_features,test_labels=joblib.load("pre_test_labels_test")
print accuracy_score(test_labels, predictions)


predictions,test_features,test_labels=joblib.load("pre_test_labels_train_SGD")
print accuracy_score(test_labels, predictions)

predictions,test_features,test_labels=joblib.load("pre_test_labels_test_SGD")
print accuracy_score(test_labels, predictions)



