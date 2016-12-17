#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

# linear
# training time: 234.423s  

# rbf, C=10000
# training time: 141.445 s

# linear
# prediction time: 21.846s

# rbf, C=10000
# prediction time: 14.378 s

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train, labels_train) 

t0 = time()
pred = clf.predict(features_test)

ten = pred[10]
twosix = pred[26]
fifty = pred[50]

print ten
print twosix
print fifty

print accuracy_score(labels_test, pred)
