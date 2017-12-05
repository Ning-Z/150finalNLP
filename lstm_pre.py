import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from data_pre import *
from lstm import *

def test_lstm(sent ,prediction):
	sent = np.array(sent_to_vec(sent[:-1], lexicon)).reshape((-1, n_chunks, chunk_size))
	with tf.Session() as sess:
		saver.restore(sess, 'model.ckpt')
		p = tf.argmax(prediction,1)
		pre = p.eval(feed_dict={x:sent}, session=sess)
		print(pre)
	return pre

# sent = ['to', 'contribute', 'any', 'really', 'good', 'enigmas']
sent = ['.--', 'Fancying', 'you', 'to', 'have', 'fathomed']
pre = test_lstm(sent, predict)
print(pre)
print(gre[test_lstm(sent, predict)[0]])