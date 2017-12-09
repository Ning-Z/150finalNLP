import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from data_pre import *
from lstm import *
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as stemmer
import sys

def test_lstm(sent ,prediction):
	sent = np.array(sent_to_vec(sent[:-1], lexicon)).reshape((-1, n_chunks, chunk_size))
	with tf.Session() as sess:
		saver.restore(sess, 'model.ckpt')
		p = tf.argmax(prediction,1)
		pre = p.eval(feed_dict={x:sent}, session=sess)
	return pre

# sent = ['to', 'contribute', 'any', 'really', 'good', 'enigmas']
# sent = ['.--', 'Fancying', 'you', 'to', 'have', 'fathomed']
# print(gre[test_lstm(sent, predict)[0]])


input = "All children, except one, grow up. They soon know that they will grow\
up, and the way Wendy knew was this. One day when she was two years old\
she was playing in a garden, and she plucked another flower and ran with\
it to her mother. I suppose she must have looked rather delightful, for\
Mrs. Darling put her hand to her heart and cried, “Oh, why can’t you\
remain like this for ever!” This was all that passed between them on\
the subject, but henceforth Wendy knew that she must grow up. You always\
know after you are two. Two is the beginning of the end."


stem = stemmer("english")
def changeable():
	# read in the changeable word vocabulary
	return set(line.strip('\n') for line in open('synonym.txt'))

change = changeable()
def replacable(word):
	word = stem.stem(word)
	if word in change:
		return True
	else: return False

def complifier2():
	input = "Of course they lived at 14 [their house number on their street], and\
until Wendy came her mother was the chief one. She was a lovely lady,\
with a romantic mind and such a sweet mocking mouth. Her romantic\
mind was like the tiny boxes, one within the other, that come from the\
puzzling East, however many you discover there is always one more; and\
her sweet mocking mouth had one kiss on it that Wendy could never get,\
though there it was, perfectly conspicuous in the right-hand corner."
	sents = ""
	output = ""
	for line in input:
		sents += line.replace('\n',' ')
	sents = sent_tokenize(sents)
	for sent in sents:
		sent = word_tokenize(sent)
		for word in sent:
			if replacable(word):
				i = sent.index(word)
				sixgram = []
				count = 5
				while count > 0:
					if sent[i-count]:
						sixgram.append(sent[i-count])
					else: sixgram.append('un_k')
					count = count - 1
				sixgram.append(word)
				replaceword = gre[test_lstm(sixgram, predict)[0]]
				output += '<div class="mytooltip">'+word+'<span class="mytooltiptext">'+replaceword+'</span></div>'
					# add something to implement the tooltip
			else:
				# print(word)
				output = output + word + " "
				# add something to implement to tooltip
	return output

# def test_argu(sys.argv[1]):
# 	print(sys.argv[1])
# print(complifier2(sys.argv[1]))
output2 = complifier2()
# if __name__=='__main__':
# 	sys.exit(main(sys.argv[1]))