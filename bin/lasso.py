#!/bin/python

# LASSO Gradient loss functions

from keras.losses import mae
from keras import backend as K


def get_lwg(grads, alpha=0.01):
    def loss_with_grad(y_true, y_pred, alpha=alpha):
        bce = mae(y_true, y_pred)
        grad_penalty = K.mean(K.sum(K.abs(grads), axis=(1, 2)))
        return bce + alpha * grad_penalty
    return loss_with_grad

def get_gp(grads, alpha=0.01):
    def grad_penalty(y_true, y_pred, alpha=alpha):
        grad_penalty = K.mean(K.sum(K.abs(grads), axis=(1, 2)))
        return alpha * grad_penalty
    return grad_penalty
