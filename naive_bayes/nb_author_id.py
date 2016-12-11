#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features and labels training and testing data
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print accuracy 
