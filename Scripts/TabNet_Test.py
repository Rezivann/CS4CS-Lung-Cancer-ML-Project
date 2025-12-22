
#import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
import torch.nn as nn

import os

from pytorch_tabnet import TabNetClassifier
import torch
import skorch
from skorch import NeuralNetClassifier
# Importing sklearn and TSNE.
import sklearn

feature_names = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring', 'Level']

df = pd.read_csv('datasets/Processed_set.csv')

X = df.drop(['Level'], axis=1)
y = df['Level']

sum = 0

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)
    #X_val, X_test, y_val, y_test = train_test_split(X_evaluate, y_evaluate, test_size=.5, shuffle=True)
    npX_train = X_train.values
    #npX_val = X_val.values
    npX_test = X_test.values
    #npX_evaluate = X_evaluate.values
    npy_train = y_train.values
    #npy_val = y_val.values
    npy_test = y_test.values
    #npY_evaluate = y_evaluate.values

    '''
    TorchX_train = torch.from_numpy(npX_train)
    Torchy_train = torch.from_numpy(npy_train)
    '''

    clf = TabNetClassifier(n_a=22, n_d=8, n_steps=2, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=0.0375))
    clf.fit(npX_train, npy_train, batch_size=100, virtual_batch_size=10, max_epochs=200, eval_metric = ['accuracy'], compute_importance=True) #eval_set=[(npX_val, npy_val)]
    #featureimportances = clf.feature_importances_
    print(npy_test)
    preds = clf.predict(npX_test)

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=preds)
    print(accuracy)
    sum += accuracy

average = sum/10
print(average)

'''
for i in range(23):
    print(f"{feature_names[i]}: {featureimportances[i]}")
'''

#print("Predictions:", preds)
