import numpy as np
import pylab as plt
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')  # noqa

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf

from zinb import ZINB


def random_nbinom(mean, disp, n=None):
    eps = 1e-10
    # translate these into gamma parameters
    gamma_shape = 1 / (disp+eps)
    gamma_scale = (mean / (gamma_shape+eps))+eps
    gamma_samples = np.random.gamma(gamma_shape, gamma_scale, n)
    return np.random.poisson(gamma_samples)


# DATA
nb_mean = 4
nb_disp = 5.5
nb_numsample = 1000
nb_samples = random_nbinom(nb_mean, nb_disp, nb_numsample)

num_sample = 1000
num_feat = 10
num_out = 10
dispersion = np.random.uniform(0, 2, [1, num_out])

X = np.random.normal(0, 0.5, (num_sample, num_feat)).astype(np.float32)
W = np.random.normal(0, 0.5, (num_feat, num_out)).astype(np.float32)
b = np.random.normal(0, 0.5, (1, num_out)).astype(np.float32)
y_mean = np.exp(np.dot(X, W) + b)
y_disp = np.zeros_like(y_mean) + dispersion

Y = random_nbinom(mean=y_mean, disp=y_disp)
print(Y[:5])

# MODEL
inputs = Input(shape=(num_feat,))
predictions = Dense(num_out, activation=tf.exp)(inputs)
model = Model(inputs=inputs, outputs=predictions)

pi_layer = Dense(num_out, activation='sigmoid')
pi = pi_layer(inputs)
zinb = ZINB(pi, theta_init=tf.zeros([1, num_out]))

model.layers[-1].trainable_weights.extend([zinb.theta_variable,
                                           *pi_layer.trainable_weights])

opt = RMSprop(lr=5e-2)
model.compile(optimizer=opt,
              loss=zinb.loss)  # zinb loss function

early_stop = EarlyStopping(monitor='val_loss', patience=20)
tb = TensorBoard(log_dir='./logs_zinb', histogram_freq=1)

losses = model.fit(X, Y,
                   callbacks=[early_stop, tb],
                   batch_size=128,
                   validation_split=0.2,
                   epochs=10000, verbose=0)

val_hist = losses.history['val_loss']
train_hist = losses.history['loss']

plt.plot(range(len(val_hist)), val_hist, 'b.-',
         range(len(train_hist)), train_hist, 'g.-')
plt.ylabel('Loss')
_ = plt.xlabel('Steps')


def visualize(x, y, x_label, y_label):
    plt.figure()
    x, y = x.squeeze(), y.squeeze()
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [(min(x)), max(x)], "g--")  # poor man's abline
    plt.xlabel(x_label)
    plt.ylabel(y_label)


x = K.eval(model.layers[-1].kernel).reshape(-1)
y = W.reshape(-1)
visualize(x, y, 'Estimated weight parameters', 'True weight parameters')

x = K.eval(model.layers[-1].bias).reshape(-1)
y = b.reshape(-1)
visualize(x, y, 'Estimated bias parameters', 'True bias parameters')

x = K.eval(zinb.theta)
y = dispersion
visualize(x, y, 'Estimated theta params', 'True theta params')

pi_model = Model(inputs=inputs, outputs=pi)
plt.figure()
sns.distplot(pi_model.predict(X).reshape(-1), kde=False)
_ = plt.xlabel('$\pi_i$ distribution of the data')
plt.show()
