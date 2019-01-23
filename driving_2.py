from __future__ import print_function
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
import tensorflow as tf
pl.rcParams['figure.figsize'] = (8.0, 8.0)
N = 1000
IT = 100
alp = 0.0001
X = np.array([np.linspace(-2, 2, N), np.linspace(-9,7,N)])
X += nr.randn(2, N)
x, y = X
x_with_bias = np.array([(1., a) for a in x]).astype(np.float32)
losses = []
with tf.Session() as sess:
    input = tf.constant(x_with_bias)
    target = tf.constant(np.transpose([y]).astype(np.float32))
    weights = tf.Variable(tf.random_normal([2,1], 0, 0.1))
    tf.global_variables_initializer().run()
    yhat = tf.matmul(input, weights)
    yerror = tf.subtract(yhat, target)
    loss = tf.nn.l2_loss(yerror)
    update_weights = tf.train.GradientDescentOptimizer(alp).minimize(loss)
    for _ in range(IT):
        sess.run(update_weights)
        losses.append(loss.eval())
        print(weights.eval())
    beta = weights.eval()
    yhat = yhat.eval()
pl.plot(x, y, 'go', label = 'data')
f = lambda x: beta[0] + beta[1] * x;
x = np.linspace(-4,4,N)
pl.plot(x, f(x), 'r')
pl.show()
        

        
