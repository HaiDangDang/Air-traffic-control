import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt


import numpy as np
import imageio
import kalman
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from mpl_toolkits.basemap import Basemap
from collections import Counter
from math import sin, cos, sqrt, atan2, radians
from mlat import geodesy
from sensors_use_train import sensor_use

from mlat import geodesy, constants

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

import random
import tensorflow as tf
tf.__version__
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
x_train = []
x_test = []
for i in np.arange(1) + 1:
    x = np.load(f"train/TDOA/layer_2/{i}.npy")
    x = random.sample(list(x), len(x))
    if len(x_train) == 0:
        x_train = x[:int(len(x)*0.999)]
    else:
        x_train = np.append(x_train, x[:int(len(x)*0.999)], axis=0)
    print(i)
    del x
    # np.save(f"train/sen_pair/tmp_train_2.npy", x_train)
    # np.save(f"train/sen_pair/tmp_test_2.npy", x_test)
x_train = np.array(x_train)
x_test = x_train[int(len(x_train)*0.9):]
x_train = x_train[:int(len(x_train)*0.9)]

y_train = x_train[:,:3]
x_train = x_train[:,3:]
y_test = x_test[:,:3]
x_test = x_test[:,3:]

# x_test = np.load(f"train/sen_pair/small/{10}_{2}.npy")
# y_test = x_test[:,0]
# x_test = x_test[:,2:]
0.26 * 1.2

scaler = MinMaxScaler().fit(x_train)
data = scaler.transform(x_train)
test= scaler.transform(x_test)

x_val = x_test[:int(len(x_test)*0.9)]
y_val = y_test[:int(len(x_test)*0.9)]

x_test = x_test[int(len(x_test)*0.9):]
y_test = y_test[int(len(y_test)*0.9):]

# x_train, x_test, y_train, y_test = train_test_split(train, y_train, test_size=0.14, random_state=21318)
# sc = StandardScaler()
# x_train =sc.fit_transform(x_train)
# x_test =sc.transform(x_test)
# to make this notebook's output stable across runs



# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

n_inputs = x_train.shape[1]
n_neurons = 10
n_outputs = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None, n_outputs])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
he_init = tf.contrib.layers.xavier_initializer()

o_input = tf.layers.dense(X, n_inputs, activation=tf.nn.elu)
o_input = tf.layers.dropout(o_input, rate=0.5, training=tf_is_training)  # drop out 50% of inputs

o_hidden_1 = tf.layers.dense(o_input, n_neurons, activation=tf.nn.elu, kernel_initializer=he_init)
o_hidden_1 = tf.layers.dropout(o_hidden_1, rate=0.5, training=tf_is_training)  # drop out 50% of inputs

o_hidden_2 = tf.layers.dense(o_hidden_1, n_neurons, activation=tf.nn.elu, kernel_initializer=he_init)
o_hidden_2 = tf.layers.dropout(o_hidden_2, rate=0.5, training=tf_is_training)  # drop out 50% of input

logits = tf.layers.dense(o_hidden_2, n_outputs)
y_proba = tf.nn.sigmoid(logits)

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))
#accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred_correct, tf.float32))))

init = tf.global_variables_initializer()

n_ecpochs = 10
batch_size = 1000


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


with tf.Session() as sess:
    init.run()
    for epoch in range(n_ecpochs):
        for iteration in range(len(x_train) // batch_size):
            X_batch, y_batch = random_batch(x_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, tf_is_training: True})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, tf_is_training: False})

        acc_val = accuracy.eval(feed_dict={X: x_val, y: y_val, tf_is_training: False})
        y_proba_val = y_proba.eval(feed_dict={X: x_val, y: y_val, tf_is_training: False})
        print(epoch, "Train accuracy:", acc_train, 'Validation accuracy:', acc_val)
with tf.Session() as sess:

    #acc_test = accuracy.eval(feed_dict={X: data_test_a})
    y_proba_test = y_proba.eval(feed_dict={X: x_test, tf_is_training: False})
    # # print(epoch, "Test prob:", y_proba_test)

    x = tf.constant([1.8, 2.2], dtype=tf.float32)

from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(n_inputs,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(128, activation="relu", name="dense_2")(x)
x = layers.Dense(256, activation="relu", name="dense_3")(x)
x = layers.Dense(128, activation="relu", name="dense_4")(x)
x = layers.Dense(64, activation="relu", name="dense_5")(x)

outputs = layers.Dense(n_outputs, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.RootMeanSquaredError()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

model.predict(x_test[:3])