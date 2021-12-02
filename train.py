import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys

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

inputs = keras.Input(shape=(9, 9, 2))
x = inputs
x = layers.Conv2D(18, 3, strides=(1,1))(x)
x = layers.Activation(keras.activations.relu)(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(81, activation="softmax")(x)
outputs = layers.Reshape((9, 9))(x)
model = keras.Model(inputs, outputs)



#def move_accuracy(y_true, y_pred):
#  y_pred = tf.round(y_pred)
#  comp = tf.math.logical_and(tf.math.equal(y_pred, 1.0), tf.math.equal(y_true, 1.0))
#  return tf.math.count_nonzero(comp) / tf.math.count_nonzero(tf.math.equal(y_true, 1.0))

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
  

model.compile(optimizer=keras.optimizers.Adam(1e-2),
              loss=move_loss,
              metrics=[move_accuracy])

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

aug_train_inputs = []
aug_train_outputs = []
for i, x in enumerate(train_inputs):
  aug_train_inputs.append(np.fliplr(x))
  aug_train_outputs.append(np.fliplr(train_outputs[i]))

train_inputs = np.concatenate((train_inputs, np.array(aug_train_inputs)))
train_outputs = np.concatenate((train_outputs, np.array(aug_train_outputs)))

aug_train_inputs = []
aug_train_outputs = []
for i, x in enumerate(train_inputs):
  aug_train_inputs.append(np.transpose(x, axes=[1, 0, 2]))
  aug_train_outputs.append(np.transpose(train_outputs[i], axes=[1, 0]))

train_inputs = np.concatenate((train_inputs, np.array(aug_train_inputs)))
train_outputs = np.concatenate((train_outputs, np.array(aug_train_outputs)))
print("augmented train set size: " + str(train_inputs.shape[0]))

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

print("filtered valid set size: " + str(valid_inputs.shape[0]))

model.fit(train_inputs, train_outputs, validation_data=valid_ds, epochs=100)

model.save(sys.argv[3])

print(train_outputs[100])
print(model.predict(np.array([train_inputs[100]])))

