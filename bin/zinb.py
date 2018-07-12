# From: https://github.com/gokceneraslan/neuralnet_countmodels/blob/master/Count%20models%20with%20neuralnets.ipynb
import tensorflow as tf


# We need a class (or closure) here,
# because it's not possible to
# pass extra arguments to Keras loss functions
# See https://github.com/fchollet/keras/issues/2121

# dispersion (theta) parameter is a scalar by default.
# scale_factor scales the nbinom mean before the
# calculation of the loss to balance the
# learning rates of theta and network weights


class NB(object):
    def __init__(self, theta=None, theta_init=[0.0],
                 scale_factor=1.0, scope='nbinom_loss/',
                 debug=False, **theta_kwargs):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope

        with tf.name_scope(self.scope):
            # a variable may be given by user or it can be created here
            if theta is None:
                theta = tf.Variable(theta_init, dtype=tf.float32,
                                    name='theta', **theta_kwargs)

            # keep a reference to the variable itself
            self.theta_variable = theta

            # to keep dispersion always non-negative
            self.theta = tf.nn.softplus(theta)

    def loss(self, y_true, y_pred, reduce=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            theta = 1.0/(self.theta+eps)

            t1 = -tf.lgamma(y_true+theta+eps)
            t2 = tf.lgamma(theta+eps)
            t3 = tf.lgamma(y_true+1.0)
            t4 = -(theta * (tf.log(theta+eps)))
            t5 = -(y_true * (tf.log(y_pred+eps)))
            t6 = (theta+y_true) * tf.log(theta+y_pred+eps)

            if self.debug:
                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)
                tf.summary.histogram('t3', t3)
                tf.summary.histogram('t4', t4)
                tf.summary.histogram('t5', t5)
                tf.summary.histogram('t6', t6)

            final = t1 + t2 + t3 + t4 + t5 + t6

            if reduce:
                final = tf.reduce_mean(final)

        return final


class ZINB(NB):
    def __init__(self, pi, scope='zinb_loss/', pi_reg=0.1,
                 **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.pi_reg = pi_reg

    def loss(self, y_true, y_pred):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            nb_case = super().loss(y_true, y_pred, reduce=False) \
                      - tf.log(1.0-self.pi+eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = 1.0/(self.theta+eps)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            result = tf.reduce_mean(result)
            result = result + self.pi_reg * tf.reduce_mean(self.pi)  # reg

            if self.debug:
                tf.summary.histogram('nb_case', nb_case)
                tf.summary.histogram('zero_nb', zero_nb)
                tf.summary.histogram('zero_case', zero_case)

        return result
