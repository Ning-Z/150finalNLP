import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from data_pre import *
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


# length of lexicon is
hm_epochs = 50
n_classes = len(gre)
batch_size = 100
chunk_size = len(lexicon)
n_chunks = 5
# rnn_size = len(lexicon)
rnn_size = 128
# input
x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')

def lstm_model(x):
	# U = {'weights': tf.Variable(tf.random_normal([number_voc, hidden_nodes])),\
	#    'biases': tf.Variable(tf.random_normal([hidden_nodes]))}
	# V = {'weights': tf.Variable(tf.random_normal([hidden_nodes, class_size])),\
	#    'biases': tf.Variable(tf.random_normal([class_size]))}
	# hidden = tf.add(tf.matnul(data, U['weights']), U['biases'])
	# hidden = tf.nn.relu(hidden)
	# output = tf.matnul(hidden, V['weights']) + V['biases']
	
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

	print('Finished building LSTM model')
	return output

def train_neural_network(x, prediction):
	f_set, lexicon = main2()
	train_x,train_y,test_x,test_y = split(f_set)
	print("Splitting input into train and test finished")
	print(len(train_x))
	print('Building up LSTM model')

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

	# learning rate default 0.001
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		# sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())

		print('Initialize_all_variables started to train:')

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# for _ in range(int(mnist.train.num_examples/batch_size)):
			i = 0
			while i < len(train_x):
				start = i
				end  = i + batch_size
				# epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				# epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

				epoch_x = np.array(train_x[start:end])
				epoch_x = epoch_x.reshape(batch_size,n_chunks,chunk_size)
				epoch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer,loss], feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch:', epoch+1, ' Completed out of:', hm_epochs, ' Loss:', epoch_loss)

		saver.save(sess, './model.ckpt')
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:np.array(test_x).reshape((-1, n_chunks, chunk_size)), y:test_y}))
		# print('Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


predict = lstm_model(x)
saver = tf.train.Saver()
# Only need to be trained once and then load the model with test
# train_neural_network(x, predict)
