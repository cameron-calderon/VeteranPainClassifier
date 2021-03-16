#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:13:24 2021

@author: CameronCalderon
"""
import sys
import csv
import statistics
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def entropyFunc(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

#Argument Handler
arg = sys.argv[1]
dataType = ['BP Dia_mmHg' , 'EDA_microsiemens' , 'LA Systolic BP_mmHg' , 'Respiration Rate_BPM']
if arg == "dia":
    arg=dataType[0]
elif arg == "eda":
    arg=dataType[1]
elif arg == "sys":
    arg=dataType[2]
elif arg == "res":
    arg=dataType[3]
elif arg == "all":
    arg= "All"
    
meanData=[]
varianceData=[]
entropyData=[]
maxData=[]
minData=[]
rfTarget=[]

df = pd.read_csv('Project1Data.csv', header=None, sep='\n')
df = df[0].str.split('\s\|\s', expand=True)
for i in range(len(df)):
    list_ob = list(df.iloc[i])
    list_ob = list_ob[0].split(',')
    new_list = [float(i) for i in list_ob[3:]]
    if (list_ob[1]==arg) or (arg=="All"):
        meanData.append(statistics.mean(new_list))
        varianceData.append(statistics.pvariance(new_list))
        entropyData.append(entropyFunc(new_list))
        maxData.append(max(new_list))
        minData.append(min(new_list))
        rfTarget.append(list_ob[2])
        

data = [list(x) for x in zip(meanData, varianceData, entropyData, maxData, minData)]
plt.plot(data)
plt.boxplot(meanData)  
plt.show() 
plt.boxplot(varianceData)  
plt.show() 
plt.boxplot(entropyData)  
plt.show() 
plt.boxplot(maxData)  
plt.show() 
plt.boxplot(minData)  
plt.show() 

#Random Forest
cm=0
X = data
y = rfTarget
kf = KFold(n_splits=10)
kf.get_n_splits(X)
rf = RandomForestClassifier(max_depth=10, random_state=0)
for train_index, test_index in kf.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    rf.fit(X_train, y_train)
    rfPre = rf.predict(X_test)
    cm += confusion_matrix(y_test, rfPre)
    print('Random Forest with 5 features (', 10, '):\n', cm)
    target =["yes", "no"]
    print(classification_report(y_test, rfPre, target_names=target))
    
print("Average: ")
print(cm/10)
