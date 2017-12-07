import tensorflow as tf
import numpy as np
from matplotlib.dates import num2date
from numpy.ma.core import minimum
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

def map_years(input):
    for i in range(len(input)):
        input[i]=input[i]-1922

def normalize_x(input):
    for i in range(0,len(input)):
        minimum=min(input[i])
        maximum=max(input[i])
        for j in range(0,len(input[i])):
            input[i][j]=(input[i][j]-minimum)/(maximum-minimum)

def next_batch(batch_size,data,labels):
    idx=np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:batch_size]
    data_shuffle=data[idx]
    labels_shuffle=labels[idx]
    return data_shuffle, labels_shuffle

num_Classses=90

xy=np.loadtxt('year_prediction_2.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,1:]
y_data=xy[:,[0]]
normalize_x(x_data)
map_years(y_data)
train_data=x_data[0:463715]
#normalize_x(train_data)
train_label=np.array(y_data[0:463715])
train_label=tf.one_hot(train_label,num_Classses)
train_label=tf.reshape(train_label,[-1,num_Classses])
train_label=train_label.eval(session=tf.Session())

#normalize_y(train_label)
test_data=np.array(x_data[0:51630])
#normalize_x(test_data)
test_label=np.array(y_data[0:51630])
test_label=tf.one_hot(test_label,num_Classses)
test_label=tf.reshape(test_label,[-1,num_Classses])
test_label=test_label.eval(session=tf.Session())
#normalize_y(test_label)

X=tf.placeholder(tf.float32,shape=[None,12])
Y=tf.placeholder(tf.float32,shape=[None,num_Classses])
keep_prob=tf.placeholder(tf.float32)


W1=tf.get_variable("W1",[12,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([128]))
layer1=tf.tanh(tf.matmul(X,W1)+b1)
layer1=tf.nn.dropout(layer1,keep_prob=keep_prob)

W2=tf.get_variable("W2",[128,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([128]))
layer2=tf.tanh(tf.matmul(layer1,W2)+b2)
layer2=tf.nn.dropout(layer2,keep_prob=keep_prob)

W3=tf.get_variable("W3",[128,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([128]))
layer3=tf.tanh(tf.matmul(layer2,W3)+b3)
layer3=tf.nn.dropout(layer3,keep_prob=keep_prob)

W4=tf.get_variable("W4",[128,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([128]))
layer4=tf.tanh(tf.matmul(layer3,W4)+b4)
layer4=tf.nn.dropout(layer3,keep_prob=keep_prob)

W5=tf.get_variable("W5",[128,num_Classses],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([num_Classses]))
hypothesis=tf.nn.softmax(tf.matmul(layer4,W5)+b5)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
train=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#recall=tf.metrics.recall(labels=Y,predictions=hypothesis)
is_correct=tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))
training_epochs=100
batch_size=1000



print("Start learning!")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#recall initalize 용으로 test
    for epoch in range(training_epochs):
        #avg_cost=0
        #total_batch = int(463715 / batch_size)
        #for i in range(total_batch):
        #    x_batch,y_batch=next_batch(batch_size,train_data,train_label)
        #    c, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
        #    avg_cost += c/total_batch
        c,_=sess.run([cost,train],feed_dict={X:train_data,Y:train_label,keep_prob:0.5})
        print('Epoch: ', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(c))
    print("Accuracy: ",sess.run(accuracy,feed_dict={X:test_data,Y:test_label,keep_prob:1}))