import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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

if len(sys.argv) == 5:
    model = keras.models.load_model(sys.argv[4], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})
else:
  inputs = keras.Input(shape=(9, 9, 3))
  x = inputs[:,:,:,0:2]
  #x = inputs
  x1 = layers.Conv2D(64, 3, strides=(3,3))(x)
  x1 = layers.UpSampling2D(3)(x1)
  x2 = layers.Conv2D(64, 3, padding="same")(x)
  x = layers.Concatenate(axis=3)([x1, x2])
  #print(x.shape)
  #x = layers.Activation(keras.activations.relu)(x)
  #x = layers.BatchNormalization()(x)
  #num_layers = 12
  #for i in range(num_layers):
  #  #tart_x = x
  #  x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-6))(x)
  #  x = layers.Activation(keras.activations.relu)(x)
  #  x = layers.BatchNormalization()(x)
  #  #x = layers.Dropout(0.2)(x)
  #x = layers.Conv2D(1, 1, padding="same")(x)
  #x = layers.Activation(keras.activations.relu)(x)
  #x = layers.BatchNormalization()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(500, activation="tanh")(x)
  x = layers.Dense(1, activation="tanh")(x)
  model = keras.Model(inputs, x)
  model.compile(optimizer=keras.optimizers.Adam(1e-2),
              loss="mse")
  model.summary()



#def move_accuracy(y_true, y_pred):
#  y_pred = tf.round(y_pred)
#  comp = tf.math.logical_and(tf.math.equal(y_pred, 1.0), tf.math.equal(y_true, 1.0))
#  return tf.math.count_nonzero(comp) / tf.math.count_nonzero(tf.math.equal(y_true, 1.0))

  

with open(sys.argv[1], "rb") as f:
  (train_inputs, train_outputs) = pickle.load(f)

print("pre-filtered train set size: " + str(train_inputs.shape[0]))

#aug_train_inputs = []
#aug_train_outputs = []
#for i, x in enumerate(train_inputs):
#  aug_train_inputs.append(np.flipud(x))
#  aug_train_outputs.append(np.flipud(train_outputs[i]))
#
#train_inputs = np.concatenate((train_inputs, np.array(aug_train_inputs)))
#train_outputs = np.concatenate((train_outputs, np.array(aug_train_outputs)))

#aug_train_inputs = []
#aug_train_outputs = []
#for i, x in enumerate(train_inputs):
#  aug_train_inputs.append(np.fliplr(x))
#  aug_train_outputs.append(np.fliplr(train_outputs[i]))
#
#train_inputs = np.concatenate((train_inputs, np.array(aug_train_inputs)))
#train_outputs = np.concatenate((train_outputs, np.array(aug_train_outputs)))
#
#aug_train_inputs = []
#aug_train_outputs = []
#for i, x in enumerate(train_inputs):
#  aug_train_inputs.append(np.transpose(x, axes=[1, 0, 2]))
#  aug_train_outputs.append(np.transpose(train_outputs[i], axes=[1, 0]))
#
#train_inputs = np.concatenate((train_inputs, np.array(aug_train_inputs)))
#train_outputs = np.concatenate((train_outputs, np.array(aug_train_outputs)))
#print("augmented train set size: " + str(train_inputs.shape[0]))

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
    #y[idx] = np.flip(y[idx], axis=1)
    idx = np.random.choice(np.arange(batch_size), batch_size//2, replace=False)
    x[idx] = np.flip(x[idx], axis=2)
    #y[idx] = np.flip(y[idx], axis=2)
    idx = np.random.choice(np.arange(batch_size), 3*batch_size//4, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    #y[idx] = np.rot90(y[idx], axes=(1,2))
    idx = np.random.choice(idx, batch_size//2, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    #y[idx] = np.rot90(y[idx], axes=(1,2))
    idx = np.random.choice(idx, batch_size//4, replace=False)
    x[idx] = np.rot90(x[idx], axes=(1,2))
    #y[idx] = np.rot90(y[idx], axes=(1,2))
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

model.fit(train_gen(), validation_data=valid_ds, epochs=100, steps_per_epoch=420)
#model.fit(train_inputs, train_outputs, validation_data=valid_ds, batch_size=batch_size,epochs=100)

model.save(sys.argv[3])

print(train_outputs[100])
print(model.predict(np.array([train_inputs[100]])))

