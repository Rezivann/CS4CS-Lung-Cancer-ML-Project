
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os


import sys 
sys.path.append("../tf_tabnet/")


#from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import tabnet_model


import tabnet_keras as tk
from tabnet_keras import TabNetClassifier

from tensorboard.plugins.hparams import api as hp
#import tensorflow_datasets as tfds
from tensorboard import version
#print(version.VERSION)




# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
import seaborn as sns






index = 0

while True:
    try:
        Name = "Test" + str(index)
        os.makedirs("C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\" + Name )
    except FileExistsError:
        index += 1
        continue
    break


feature_names = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring', 'Level']

df = pd.read_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\Processed_set.csv')
val_data = df.sample(frac = .2, random_state = 45)
train_data = val_data.drop(val_data.index)
print(train_data.dtypes)
print(train_data.nunique())
X = df.drop(['Level'], axis=1)
y = df['Level']

def dataframe_to_tf_dataset(dataframe: pd.DataFrame, target_name: str):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_name)
    return tf.data.Dataset.from_tensor_slices(
        (dict(dataframe), labels)
    )

train_ds = dataframe_to_tf_dataset(train_data, "Level")
val_ds = dataframe_to_tf_dataset(val_data, "Level")
test_ds = dataframe_to_tf_dataset(val_data, "Level") # I NEED TO TROUBLESHOOT THIS BC HE INPUTTED TWO SEPARATE CSVS AND SPLIT THE TRAINING ONE

batch_size = 80

train_ds = train_ds.shuffle(320, seed=123).batch(batch_size).prefetch(1)
for x, y in train_ds.unbatch().take(1):
    print(f"X: {x}")
    print(f"y: {y}")
    print(f"X: {x['Age']}")

feature_names.remove('Level')

cat_str_feature_names = list()
cat_int_feature_names = feature_names

cat_embed_dims = {}

def create_keras_input_layer(feature_names, cat_str_feature_names, cat_int_feature_names):
    model_inputs = list()

    for name in feature_names:
        if name in cat_str_feature_names:
            dtype = tf.string
        elif name in cat_int_feature_names:
            dtype = tf.int64
        else:
            dtype = tf.float32
        shape = (1,) if dtype==tf.float32 else ()
        model_inputs.append(tf.keras.Input(shape=shape, name=name, dtype=dtype))
    return model_inputs

def encode_categorical_feature(keras_input, feature_name, dataset, embed_dim, is_string):
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    lookup_fn = tf.keras.layers.StringLookup if is_string else tf.keras.layers.IntegerLookup
    lookup = lookup_fn(output_mode="int")
    lookup.adapt(feature_ds)
    encoded_feature = lookup(keras_input)
    embedded_feature= tf.keras.layers.Embedding(
        input_dim=lookup.vocabulary_size(),
        output_dim=embed_dim,
        name=f"{feature_name}_embedding"
    ) (encoded_feature)

    return embedded_feature

def encode_features(keras_inputs, feature_names, cat_str_feature_names, cat_int_feature_names, cat_embed_dims, dataset):
    encoded_features = list()

    for keras_input, feature_name in zip(keras_inputs, feature_names):
        if feature_name in cat_str_feature_names or feature_name in cat_int_feature_names:
            embed_dim = cat_embed_dims[feature_name] if feature_name in cat_embed_dims.keys() else 1
            encoded_features.append(
                encode_categorical_feature(keras_input, feature_name, dataset, embed_dim, feature_name in cat_str_feature_names)
            )
        else:
            encoded_features.append(keras_input)
    return encoded_features

tabnet_params = {
    "decision_dim": 16,
    "attention_dim": 16,
    "n_steps": 5,
    "n_shared_glus" : 2, 
    "n_dependent_glus": 2, 
    "relaxation_factor": 1.5,
    "epsilon": 1e-15,
    "virtual_batch_size": None,
    "momentum": 0.98,
    "mask_type": "entmax",
    "lambda_sparse": 1e-4
}

inputs = create_keras_input_layer(feature_names, cat_str_feature_names, cat_int_feature_names)
x = encode_features(inputs, feature_names, cat_str_feature_names, cat_int_feature_names, cat_embed_dims, train_ds)
x = tf.keras.layers.Concatenate()(x)
x = tabnet_model.TabNetEncoder(**tabnet_params)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


direct = 'C:\\Users\\reziv\\Downloads\\Lung_Cancer_Model\\' + Name

from sklearn.model_selection import train_test_split



#print(list(X.columns))

'''datasetX = tf.convert_to_tensor(X_train)
datasetY = tf.convert_to_tensor(y_train)'''


'''min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)'''
'''
names = list(X.columns)
feature_columns = []
for col_name in names:
    feature_columns.append(tf.feature_column.numeric_column(col_name))
'''


'''
tmodel = TabNetClassifier(n_classes=3, out_activation= None)
tmodel.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
datasetX = tf.convert_to_tensor(X_train)
datasetXX = tf.convert_to_tensor(y_train)

tmodel.fit(datasetX, epochs = 100, validation_data=datasetXX, verbose=1)
tmodel.summary()
'''



'''
x, _ = next(iter(datasetX))
_ = model(x)
'''





'''
(X_Train, X_Test)  = tf.keras.utils.split_dataset(
    datasetX, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
(Y_Train, Y_Test)  = tf.keras.utils.split_dataset(
    datasetY, left_size=0.8, right_size=0.2, shuffle = False, seed = None
)
'''

#features = np.array(X_train)
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
'''
def train_test_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(1, (23), activation='relu'), # input_shape=(32, 32, 3)
        tf.keras.layers.MaxPooling1D((2, 2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
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

def run(run_dir):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

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