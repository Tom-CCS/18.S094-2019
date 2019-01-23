import os
import tensorflow as tf
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
from six.moves.urllib.request import urlretrieve

import gzip, binascii, struct, numpy

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"

def maybe_download(filename):
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

IMAGE_SIZE = 28
PIXEL_DEPTH = 255
#Normalizing is important here
def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

train_data = extract_data(train_data_filename, 60000)
test_data = extract_data(test_data_filename, 10000)


NUM_LABELS = 10

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

train_labels = extract_labels(train_labels_filename, 60000)
test_labels = extract_labels(test_labels_filename, 10000)

VALIDATION_SIZE = 5000

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
validation_labels = numpy.argmax(validation_labels, 1)
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]
#Paramters
train_size = train_labels.shape[0]
BATCH_SIZE = 60
EVAL_SIZE = 10
NUM_LABEL = 10
NUM_CHANNEL = 1
LAYER1 = 32
LAYER2 = 64
FILTER = 5
SEED = 42
BATCH = 0
FC1 = 500 #512 in example
FC2 = NUM_LABEL # produces final result
TRAIN_ITER = 1000
LEARN_RATE = 0.01
offset = 0
if (True):
    #variables. 2 layers of Conv + relu and 2 layers of FC
    cur_batch = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE,  NUM_CHANNEL]) #updated with each iteration
    validation = tf.constant(validation_data) #held constant
    test = tf.constant(test_data)
    yhat = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_LABEL])                       
    conv1_weights = tf.Variable(tf.random_normal([FILTER, FILTER, NUM_CHANNEL, LAYER1], 0, 0.1))
    conv1_bias = tf.Variable(tf.zeros([LAYER1]))
    conv2_weights = tf.Variable(tf.random_normal([FILTER, FILTER, LAYER1, LAYER2], 0, 0.1))
    conv2_bias = tf.Variable(tf.zeros([LAYER2]))
    fc1_weights = tf.Variable(tf.random_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * LAYER2, FC1], 0, 0.1))
    fc2_weights = tf.Variable(tf.random_normal([FC1, FC2], 0.1, 0.02)) #sum of probability = 1
def model(data, train = False): #the neural net
    input = tf.reshape(data, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]) #-1 to denote uncertainty
    conv1 = tf.nn.conv2d(input, conv1_weights, strides = [1,1,1,1], padding = 'SAME')
    conv1_pool = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1_pool, conv1_bias))
    conv2 = tf.nn.conv2d(relu1, conv2_weights, strides = [1,1,1,1], padding = 'SAME')
    conv2_pool = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2_pool, conv2_bias))
    #tranform for fc
    relu2_res = tf.reshape(relu2, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * LAYER2])
    fc1 = tf.matmul(relu2_res, fc1_weights)
    fc1_activate = tf.math.tanh(fc1)
    #calculate result
    if (train): #dropout during training to prevent overfitting
        fc1_activate = tf.nn.dropout(fc1_activate, 0.5)
    fc2 = tf.matmul(fc1_activate,fc2_weights)
    return fc2 #result 
# supply error function
logit = model(cur_batch, train = True)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = yhat, logits = logit, dim = -1))#the result is a BATCH_SIZE vector, so need to take mean
regulizer = tf.nn.l2_loss(conv1_weights) + tf.nn.l2_loss(conv2_weights) + tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights)#L_2 norm regularization
BETA = 0.00005
error += BETA * regulizer
#training
batch = 0
training_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, 1000, 0.95, staircase = 'TRUE')  #the second argument varies through trainning
optimizer = tf.train.MomentumOptimizer(training_rate, 0.9).minimize(error)#A solution to smooth descent & fight local minima; the last argument updates by one automatically
#during optimization all variables will be automatically collected & updated
#make predictions
training_predictions = tf.nn.softmax(logit) #softmax is just e^{x_i} / sum e^{x_i}
validation_predictions = tf.nn.softmax(model(validation_data))
test_predictions = tf.nn.softmax(model(test_data))
#All right we're done. Time to run model!
ss = tf.InteractiveSession()
tf.global_variables_initializer().run()
ss.as_default()
for _ in range(TRAIN_ITER):
    batch += 1
    data = train_data[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :, :, :]
    label = train_labels[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
    input_dict = {cur_batch: data, yhat: label}
    _, e, tr, res, val= ss.run([optimizer, error, training_rate, training_predictions, validation_predictions], feed_dict = input_dict) #calculate the tensors in the first argument, substituting the placeholders in feed_dict
    #connvert training predictions into results:
    val = numpy.argmax(val, 1)
    #get result
    if (True):
        s = numpy.sum(tf.cast(tf.equal(val, validation_labels), tf.float32))
        s_sum = 0
        for i in range(VALIDATION_SIZE):
            s_sum += s[i]
        print("correct rate:", (s_sum / VALIDATION_SIZE).eval())
        
    
    
    
