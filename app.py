from flask import Flask, request, redirect, render_template, url_for
from xml.sax import saxutils as su
app = Flask(__name__)

from complifier import *
from lstm_pre import *
# import lstm_pre

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from data_pre import *
from lstm import *
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as stemmer

def test_lstm(sent ,prediction):
	sent = np.array(sent_to_vec(sent[:-1], lexicon)).reshape((-1, n_chunks, chunk_size))
	with tf.Session() as sess:
		saver.restore(sess, 'model.ckpt')
		p = tf.argmax(prediction,1)
		pre = p.eval(feed_dict={x:sent}, session=sess)
	return pre

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

def complifier2(input):
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

@app.route("/")
def index():
	# createlink = "<a href='" + url_for('create') + "'>start to complify</a>"
	# return """<html><head><title>home</title></head><body>"""+createlink+"""</body></html>"""
	link1 = url_for('complifing')
	link2 = url_for('complifing2')
	return render_template("homepage.html",link1=link1, link2=link2)

@app.route('/complifing',methods=['GET', 'POST'])
def complifing():
	if request.method == 'GET':
		link = url_for('index')
		return render_template("inputtext.html",link = link)
	elif request.method == 'POST':
		input1 = request.form['input'];
		output = tooltip_output(input1)
		replaceRotio = 0
		if output != "":
			replaceRotio = get_ReplaceRatio(input1)
		link = url_for('complifing')
		return render_template("answer.html",link = link, output=output, ratio = replaceRotio)
	else:
		return "<h2>Invalid input</h2>"

@app.route('/complifing2',methods=['GET', 'POST'])
def complifing2():
	if request.method == 'GET':
		link = url_for('index')
		return render_template("inputtext2.html",link = link)
	elif request.method == 'POST':
		# input2 = request.form['input2'];
		input2 = "Of course they lived at 14 [their house number on their street], and\
until Wendy came her mother was the chief one. She was a lovely lady,\
with a romantic mind and such a sweet mocking mouth. Her romantic\
mind was like the tiny boxes, one within the other, that come from the\
puzzling East, however many you discover there is always one more; and\
her sweet mocking mouth had one kiss on it that Wendy could never get,\
though there it was, perfectly conspicuous in the right-hand corner."
		# exec(open("./lstm_pre.py").read())
		# output2 = complifier2(input2)
		replaceRotio = 0
		if output2 != "":
			replaceRotio = get_ReplaceRatio(input2)
		link = url_for('complifing2')
		return render_template("answer2.html",link = link, output=output2, ratio = replaceRotio)
	else:
		return "<h2>Invalid input</h2>"

if __name__ == "__main__":
    app.run(debug=True)


