
# coding: utf-8

# In[1]:

import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model


# In[2]:

def write_result(result):
    result = result.reshape((result.shape[0],1))
    result = result.astype(float)
    result = result.astype(int)
    with open('testlabel.csv','w',newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        for row in result:
            writer.writerow(row)


# In[3]:

# read data
train_X = []
with open('traindata.csv','r') as f:
    reader = csv.reader(f,delimiter = ',')
    for row in reader:
        train_X.append(row)
        
train_Y = []
with open('trainlabel.csv','r') as f:
    reader = csv.reader(f,delimiter = ',')
    for row in reader:
        train_Y.append(row)
train_Y = np.array(train_Y)
train_Y = train_Y.reshape((train_Y.shape[0],))
        
test_X = []
with open('testdata.csv','r') as f:
    reader = csv.reader(f,delimiter = ',')
    for row in reader:
        test_X.append(row)


# In[4]:

# normalize data
scaler = StandardScaler()
train_X_normed = scaler.fit_transform(train_X)
test_X_normed = scaler.fit_transform(test_X)


# In[5]:

# split training data in to training set and validation set
trainX, validX, trainY, validY = train_test_split(train_X_normed, train_Y, test_size=0.1, random_state=17)


# In[10]:

# SVM
paramgrid = {'C': np.logspace(-3,3,13)}
svmcv = GridSearchCV(estimator=svm.SVC(kernel='linear',random_state=17), param_grid = paramgrid, cv=10,verbose=1)
svmcv.fit(trainX, trainY)


# In[11]:

svmcv.best_score_


# In[12]:

predY = svmcv.predict(validX)
acc = metrics.accuracy_score(validY, predY)
print ("test accuracy = " + str(acc))


# In[6]:

# Random Forest
paramgrid = {'n_estimators': np.array([1,2,3,5,7, 10, 13, 15, 20, 27, 35,50,80, 100]) }
rfcv = GridSearchCV(ensemble.RandomForestClassifier(random_state=71),paramgrid, cv=10)
rfcv.fit(trainX, trainY)
predY = rfcv.best_estimator_.predict(validX)


# In[7]:

acc = metrics.accuracy_score(validY, predY)
print('Accuracy on test-set: ' + str(acc))


# In[73]:

# adaboost
paramgrid = {'n_estimators': np.array([1,3,5,7, 10, 13, 15, 20, 33,50,80, 100]) }
adacv = GridSearchCV(ensemble.AdaBoostClassifier(random_state=7),paramgrid, cv=5)
adacv.fit(trainX, trainY)
predY = adacv.best_estimator_.predict(validX)
acc = metrics.accuracy_score(validY, predY)
print('Accuracy on test-set: ' + str(acc))


# In[114]:




# In[9]:

# make prediction and write to file
write_result(rfcv.best_estimator_.predict(test_X_normed))


# In[ ]:



