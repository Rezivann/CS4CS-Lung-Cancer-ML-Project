
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import os

from tensorboard.plugins.hparams import api as hp
#import tensorflow_datasets as tfds
from tensorboard import version
#print(version.VERSION)
index = 0

while True:
    try:
        Name = "Tab" + str(index)
        os.makedirs("C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Logs\\" + Name )
    except FileExistsError:
        index += 1
        continue
    break


df = pd.read_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Datasets\\Processed_set.csv')
X = df.drop(['Level'], axis=1)
y = df['Level']

direct = 'C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Logs\\' + Name

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)

def train_test_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer= 'Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train,
        y_train,
        epochs=38,
        steps_per_epoch = int(800/38),
    )
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy


session_num = 1
sum = 0

for i in range(10):
    run_name = "run-%d" % (i+1)
    print('--- Starting trial: %s' % run_name)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)
    datasetX = tf.convert_to_tensor(X_train)
    datasetY = tf.convert_to_tensor(y_train)
    data = train_test_model()
    sum += data
    print(data)
    
average = sum / 10
print(sum)
print(average)
'''
model = tf.keras.Sequential([
    ##layers.Dense(200, activation= 'relu'),
    layers.Dense(24, activation= 'sigmoid'),
    layers.Dense(12, activation= 'relu'),
    layers.Dense(1)
])
model.compile(loss = tf.keras.losses.MeanSquaredError(), 
              optimizer = tf.keras.optimizers.Adam(), metrics=['accuracy'])

stuff = model.fit(X_train, y_train, epochs=30, batch_size=5) #steps_per_epoch=50
#print(stuff.history)
loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(loss)
print(accuracy)
'''