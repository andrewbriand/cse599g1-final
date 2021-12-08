import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys
import random
import keras_tuner as kt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.run_functions_eagerly(True)

# One layer linear
#inputs = keras.Input(shape=(9, 9, 2))
#x = layers.Reshape((81 * 2,))(inputs)
#outputs = layers.Dense(81, activation="softmax")(x)
#outputs = layers.Reshape((9, 9))(outputs)
#model = keras.Model(inputs, outputs)

#inputs = keras.Input(shape=(9, 9, 2))
#x = layers.Reshape((81 * 2,))(inputs)
#x = layers.Dense(120, activation="relu")(x)
#outputs = layers.Dense(81, activation="softmax")(x)
#outputs = layers.Reshape((9, 9))(outputs)
#model = keras.Model(inputs, outputs)

#inputs = keras.Input(shape=(9, 9, 2))
#x = layers.Conv2D(32, 3)(inputs)
#x = layers.Conv2D(32, 3)(x)
#x = layers.Conv2D(32, 3)(x)
#x = layers.Flatten()(x)
#x = layers.Dropout(0.5)(x)
#outputs = layers.Dense(81, activation="softmax")(x)
#outputs = layers.Reshape((9, 9))(outputs)
#model = keras.Model(inputs, outputs)

#inputs = keras.Input(shape=(9, 9, 2))
#x = inputs
#x = layers.Conv2D(128, 3, padding="same")(x)
#x = layers.Activation(keras.activations.relu)(x)
#x = layers.BatchNormalization()(x)
#num_layers = 2
#for i in range(num_layers):
#  x_start = x
#  x = layers.Conv2D(128, 3, padding="same")(x)
#  x = layers.Activation(keras.activations.relu)(x)
#  x = layers.BatchNormalization()(x)
#  x = layers.Add()([x, x_start])
##x = layers.Dropout(0.5)(x)
#x = layers.Conv2D(1, 1)(x)
#x = layers.Activation(keras.activations.relu)(x)
##x = layers.Flatten()(x)
#x = layers.Reshape((81,))(x)
#x = layers.Dense(81, activation="softmax")(x)
#x = layers.Reshape((9, 9))(x)
#model = keras.Model(inputs, x)

#inputs = keras.Input(shape=(9, 9, 2))
#x = inputs
##x = layers.GaussianNoise(0.5)(x)
#x = layers.Conv2D(64, 3, strides=(3,3))(x)
#x = layers.Activation(keras.activations.relu)(x)
#x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.3)(x)
#num_layers = 1
#for i in range(num_layers):
#  x = layers.Conv2D(64, 3, padding='same')(x)
#  x = layers.Activation(keras.activations.relu)(x)
#  x = layers.BatchNormalization()(x)
#  x = layers.Dropout(0.3)(x)
#x = layers.Conv2D(9, 1)(x)
#x = layers.Activation(keras.activations.softmax)(x)
##x = layers.Flatten()(x)
##outputs = layers.Dense(81, activation="softmax")(x)
#outputs = layers.Reshape((9, 9))(x)
#model = keras.Model(inputs, x)

#inputs = keras.Input(shape=(9, 9, 2))
#x = inputs
##x = layers.GaussianNoise(0.5)(x)
#x = layers.Conv2D(32, 3, strides=(3,3), kernel_regularizer=keras.regularizers.l2(5e-4))(x)
#x = layers.Activation(keras.activations.relu)(x)
#x = layers.BatchNormalization()(x)
#x = layers.Flatten()(x)
#num_layers = 3
#for i in range(num_layers):
#  x = layers.Dense(81, kernel_regularizer=keras.regularizers.l2(5e-3))(x)
#  x = layers.Activation(keras.activations.relu)(x)
#  x = layers.BatchNormalization()(x)
#  x = layers.Dropout(0.5)(x)
#x = layers.Dense(81, activation="softmax", kernel_regularizer=keras.regularizers.l2(5e-3))(x)
#x = layers.Reshape((9, 9))(x)
#model = keras.Model(inputs, x)

def move_accuracy(y_true, y_pred):
  y_pred = tf.reshape(y_pred, (-1, 81))
  y_true = tf.reshape(y_true, (-1, 81))
  y_pred_idx = tf.argmax(y_pred, axis=1)
  y_true_idx = tf.argmax(y_true, axis=1)
  comp = tf.math.equal(y_pred_idx, y_true_idx)
  return tf.math.count_nonzero(comp) / tf.size(y_true_idx, out_type=tf.dtypes.int64)

def move_loss(y_true, y_pred):
  y_pred = tf.reshape(y_pred, (-1, 81))
  y_true = tf.reshape(y_true, (-1, 81))
  cce = tf.keras.losses.CategoricalCrossentropy()
  return cce(y_true, y_pred)

def create_model(hp):
  inputs = keras.Input(shape=(9, 9, 2))
  x = inputs
  x1 = layers.Conv2D(hp.Choice('s_filters_init', [8, 16, 32]), 3, strides=(3,3))(x)
  x1 = layers.UpSampling2D(3)(x1)
  x2 = layers.Conv2D(hp.Choice('filters_init', [8, 16, 32]), 3, padding="same")(x)
  x = layers.Concatenate(axis=3)([x1, x2])
  print(x.shape)
  x = layers.Activation(keras.activations.relu)(x)
  x = layers.BatchNormalization()(x)
  num_layers = hp.Choice('num_layers', [2, 4, 6, 8, 10])
  filters_stride = hp.Choice('filters_stride', [8, 16])
  filters = hp.Choice('filters', [8, 16])
  filters_1x1 = hp.Choice('filters_1x1', [8, 16])
  reg = hp.Choice('reg', [1e-6, 1e-5, 1e-4, 1e-3])
  for i in range(num_layers):
    start_x = x
    x1 = layers.Conv2D(filters_stride, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
    x2 = layers.Conv2D(filters, 3, strides=(3, 3), kernel_regularizer=keras.regularizers.l2(reg))(x)
    x2 = layers.UpSampling2D(3)(x2)
    x = layers.Concatenate(axis=3)([x1, x2])
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters_1x1, 1, kernel_regularizer=keras.regularizers.l2(reg))(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.BatchNormalization()(x)
    if i > 0:
      x = layers.Add()([x, start_x])
    #x = layers.Dropout(0.2)(x)
  x = layers.Conv2D(1, 1, padding="same")(x)
  x = layers.Activation(keras.activations.relu)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(81, activation="softmax")(x)
  x = layers.Reshape((9, 9))(x)
  model = keras.Model(inputs, x)
  lr = hp.Choice('lr', [1e-5, 1e-4, 1e-3, 1e-2])
  model.compile(optimizer=keras.optimizers.Adam(lr),
              loss=move_loss,
              metrics=[move_accuracy])
  tf.keras.backend.clear_session()
  return model

with open(sys.argv[1], "rb") as f:
  (train_inputs, train_outputs) = pickle.load(f)

print("pre-filtered train set size: " + str(train_inputs.shape[0]))


train_set = set()
new_train_inputs = []
new_train_outputs = []
train_inputs.flags.writeable = False
train_outputs.flags.writeable = False
for i, x in enumerate(train_inputs):
  if (x.tobytes(), train_outputs[i].tobytes()) not in train_set:
    train_set.add((x.tobytes(), train_outputs[i].tobytes()))
    new_train_inputs.append(x)
    new_train_outputs.append(train_outputs[i])

train_inputs = np.array(new_train_inputs)
train_outputs = np.array(new_train_outputs)

print("filtered train set size: " + str(train_inputs.shape[0]))

with open(sys.argv[2], "rb") as f:
  (valid_inputs, valid_outputs) = pickle.load(f)

print("pre-filtered valid set size: " + str(valid_inputs.shape[0]))

valid_set = set()
new_valid_inputs = []
new_valid_outputs = []
valid_inputs.flags.writeable = False
valid_outputs.flags.writeable = False
for i, x in enumerate(valid_inputs):
  if (x.tobytes(), valid_outputs[i].tobytes()) not in valid_set and (x.tobytes(), valid_outputs[i].tobytes()) not in train_set:
    valid_set.add((x.tobytes(), valid_outputs[i].tobytes()))
    new_valid_inputs.append(x)
    new_valid_outputs.append(valid_outputs[i])

valid_inputs = np.array(new_valid_inputs)
valid_outputs = np.array(new_valid_outputs)

valid_ds = (valid_inputs, valid_outputs)

batch_size = 32*15
def train_gen():
  n = 0 
  available = train_inputs.shape[0]
  while True:

    #idx = np.random.choice(np.arange(train_inputs.shape[0]), batch_size, replace=False)
    x = train_inputs[n:n+batch_size]
    y = train_outputs[n:n+batch_size]
    idx = np.random.choice(np.arange(batch_size), batch_size//2, replace=False)
    x[idx] = np.flip(x[idx], axis=1)
    y[idx] = np.flip(y[idx], axis=1)
    idx = np.random.choice(np.arange(batch_size), batch_size//2, replace=False)
    x[idx] = np.flip(x[idx], axis=2)
    y[idx] = np.flip(y[idx], axis=2)
    idx = np.random.choice(np.arange(batch_size), 3*batch_size//4, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    y[idx] = np.rot90(y[idx], axes=(1,2))
    idx = np.random.choice(idx, batch_size//2, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    y[idx] = np.rot90(y[idx], axes=(1,2))
    idx = np.random.choice(idx, batch_size//4, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    y[idx] = np.rot90(y[idx], axes=(1,2))
    yield x, y
    n += batch_size
    n = n % (available - batch_size)

#def train_gen():
#    class Iterable(object):
#      def __iter__(self):
#        n = train_inputs.shape[0]
#        for i in range(n//batch_size):
#          idx = np.random.choice(np.arange(train_inputs.shape[0]), batch_size, replace=True)
#          yield train_inputs[idx], train_outputs[idx]
#    return Iterable()

print("filtered valid set size: " + str(valid_inputs.shape[0]))

#model.fit(train_gen(), validation_data=valid_ds, epochs=100, steps_per_epoch=319)
#model.fit(train_inputs, train_outputs, validation_data=valid_ds, batch_size=batch_size,epochs=100)

tuner = kt.BayesianOptimization(create_model, objective=kt.Objective('val_move_accuracy', 'max'), max_trials=20, project_name="models/hypertune3")

tuner.search(train_gen(), epochs=20, validation_data=valid_ds, steps_per_epoch=319)

tuner.get_best_models()[0].save(sys.argv[3])
#model.save(sys.argv[3])
#
#print(train_outputs[100])
#print(model.predict(np.array([train_inputs[100]])))

