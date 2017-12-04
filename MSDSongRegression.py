import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

xy=np.loadtxt('year_prediction_2.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,1:-1]
y_data=xy[:,[0]] # Wine dataset은 class 분류가 맨 앞 column에 나와 있으므로 y_data를 맨 앞 column의 값을 잘라 온 것으로 한다.
train_data=x_data[463715:]
train_label=y_data[463715:]
test_data=x_data[51630:]
test_label=y_data[51630:]

X=tf.placeholder(tf.float32,shape=[None,12])
Y=tf.placeholder(tf.float32,shape=[None,1])
keep_prob=tf.placeholder(tf.float32)

W1=tf.get_variable("W1",[12,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([64]))
layer1=tf.nn.relu(tf.matmul(X,W1)+b1)
layer1=tf.nn.dropout(layer1,keep_prob=keep_prob)

W2=tf.get_variable("W2",[64,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([64]))
layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)
layer2=tf.nn.dropout(layer2,keep_prob=keep_prob)

W3=tf.get_variable("W3",[64,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([64]))
layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)
layer3=tf.nn.dropout(layer3,keep_prob=keep_prob)

W4=tf.get_variable("W4",[64,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([64]))
layer4=tf.nn.relu(tf.matmul(layer3,W4)+b4)
layer4=tf.nn.dropout(layer3,keep_prob=keep_prob)

W5=tf.get_variable("W5",[64,1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([1]))
hypothesis=tf.nn.relu(tf.matmul(layer4,W5)+b5)

cost=tf.reduce_mean(tf.square(hypothesis-Y))
train=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
is_correct=tf.equal(hypothesis,Y)
recall=tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))
#recall 계산식 구현!

training_epochs=50
batch_size=10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(463715/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=tf.train.batch([train_data,train_label],batch_size, allow_smaller_final_batch=True)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs.eval(), Y: batch_ys.eval(), keep_prob: 0.7})
            avg_cost += c / total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print("Recall: ",recall.eval(session=sess,feed_dict={X:test_data,Y:test_label,keep_prob:1}))