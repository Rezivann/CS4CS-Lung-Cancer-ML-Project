import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import os

from tensorboard.plugins.hparams import api as hp
#import tensorflow_datasets as tfds
from tensorboard import version
from sklearn.model_selection import train_test_split
#print(version.VERSION)
index = 0

while True:
    try:
        Name = "Multiple" + str(index)
        os.makedirs("C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\" + Name )
    except FileExistsError:
        index += 1
        continue
    break

df = pd.read_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\Processed_set.csv')
X = df.drop(['Level'], axis=1)
y = df['Level']
direct = 'C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\' + Name
nums = []
def run(run_dir, hparams, num, i):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        print(accuracy)
        if (i == 0):
            nums.append(accuracy)
        else: 
            nums[num] += accuracy
        if (i == 3):
            average = nums[num]/4
            print(average)
            tf.summary.scalar(METRIC_ACCURACY, average, step=1)
def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train,
        y_train,
        epochs=hparams[HP_EPOCHS],
        steps_per_epoch = int(800/hparams[HP_EPOCHS]),
        callbacks = [
            tf.keras.callbacks.TensorBoard(direct),
            hp.KerasCallback(direct, hparams)
        ]
    )
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy


for i in range(4):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)
    datasetX = tf.convert_to_tensor(X_train)
    datasetY = tf.convert_to_tensor(y_train)
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete(list(range(17,22, 1)))) #Was 17, 23, 1
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete(list(range(35, 41, 2)))) #Was 35, 43, 2
    METRIC_ACCURACY = 'accuracy'
    if (i == 0):
        with tf.summary.create_file_writer(direct).as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
            )
    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                for epoch in HP_EPOCHS.domain.values:
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                        HP_EPOCHS: epoch
                    }
                    run_name = "run-%d" %  i + "-" + str(session_num)
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(direct + "/" + run_name + "/", hparams, session_num, i)
                    session_num += 1
    


'''
(X_Train, X_Test)  = tf.keras.utils.split_dataset(
    datasetX, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
(Y_Train, Y_Test)  = tf.keras.utils.split_dataset(
    datasetY, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
'''



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