
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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

X_train, X_evaluate, y_train, y_evaluate = train_test_split(X,y,test_size = .2, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_evaluate, y_evaluate, test_size=.5, shuffle=True)
npX_train = X_train.values
npX_val = X_val.values
npX_test = X_test.values
npX_evaluate = X_evaluate.values
npy_train = y_train.values
npy_val = y_val.values
npy_test = y_test.values
npY_evaluate = y_evaluate.values

TorchX_train = torch.from_numpy(npX_train)
Torchy_train = torch.from_numpy(npy_train)

clf = TabNetClassifier()
print(type(X_val))
#clf.fit(npX_train, npy_train, eval_set=[(npX_val, npy_val)], max_epochs=100, eval_metric = ['accuracy'], virtual_batch_size=30)

class TabnetCustom(nn.Module):
    def __init__(self, n_d, n_a, n_steps):
        super().__init__()
        self.tabnet = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            verbose=1
        )
    def forward(self, x):
        return self.tabnet.predict_proba(x)


myTabnet = TabnetCustom(n_a=3, n_d=4, n_steps=5)
net = NeuralNetClassifier(
    module=myTabnet,
    module__n_d=8,
    module__n_a=8,
    module__n_steps=3,
    #criterion=torch.nn.CrossEntropyLoss,
    #optimizer=torch.optim.Adam,
    #optimizer__lr=0.001
    #lr=0.02,
)

#torch.compile(clf, backend="tensorrt")
#preds = clf.predict(npX_test)

#score = clf.predict_proba(npX_test)
#print(score)

param_grid = {
    'n_d': list(range(8, 9, 1)),
    'n_a': list(range(22, 23, 1)),
    'n_steps': list(range(2, 3, 1)),
    'optimizer_fn': [torch.optim.Adam, torch.optim.Adafactor, torch.optim.Adagrad, torch.optim.LBFGS, torch.optim.NAdam, torch.optim.Rprop],
    'optimizer_params': [dict(lr=0.0375), dict(lr=0.035), dict(lr=0.0325)],

}
params = {
    'optimizer': torch.optim.Adam
}

grid = sklearn.model_selection.GridSearchCV(estimator=TabNetClassifier(n_d=8, n_a=5, n_steps=4), param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=1)
#print(list(list(net.parameters())))
#optimizer = torch.optim.Adam(TabnetCustom.parameters(), lr=0.001)
grid_result = grid.fit(npX_train, npy_train, batch_size=100, virtual_batch_size = 10)
print("Best score: ", grid_result.best_score_)
print("Best params: ", grid_result.best_params_)

print()
#print("Predictions:", preds)
