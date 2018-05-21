import tensorflow as tf 
import numpy as np 

#create data
x_data = np.random.rand(100).astype(np.float64)
y_data = x_data*0.4+1
#create model
Weights=tf.Variable(tf.random_uniform([1],-1,1))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases
#error
loss=tf.reduce_mean(tf.square(y-y_data))

optimizer=tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(201):
		sess.run(train)
    	if step % 20 == 0:
        	print(step, sess.run(Weights), sess.run(biases))