
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
        os.makedirs("C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\" + Name )
    except FileExistsError:
        index += 1
        continue
    break


df = pd.read_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\Processed_set.csv')
X = df.drop(['Level'], axis=1)
y = df['Level']

direct = 'C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\' + Name

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)



datasetX = tf.convert_to_tensor(X_train)
datasetY = tf.convert_to_tensor(y_train)


'''
(X_Train, X_Test)  = tf.keras.utils.split_dataset(
    datasetX, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
(Y_Train, Y_Test)  = tf.keras.utils.split_dataset(
    datasetY, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
'''

features = np.array(X_train)
#print(features)

'''

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete(list(range(16,22, 2))))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete(list(range(38, 48, 2))))



METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer(direct).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )
'''

def train_test_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(18, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(18, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer= 'sgd',
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
'''
def run(run_dir):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
'''
session_num = 1

sum = 0
for i in range(20):
    run_name = "run-%d" % (i+1)
    print('--- Starting trial: %s' % run_name)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)
    datasetX = tf.convert_to_tensor(X_train)
    datasetY = tf.convert_to_tensor(y_train)
    data = train_test_model()
    sum += data
    print(data)
    
average = sum / 20
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