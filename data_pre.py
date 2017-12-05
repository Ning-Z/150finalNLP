from nltk.corpus import gutenberg
from nltk import ngrams
import numpy as np
from nltk.stem.snowball import SnowballStemmer as stemmer
import xlrd
import pickle
import random
from collections import Counter

def gre_voc():
	# A xls reader to get all GRE vocabularies
	gre_voc = xlrd.open_workbook("GRE-Vocab.xls").sheet_by_index(1)
	return list(set(gre_voc.cell(row, 0).value for row in range(gre_voc.nrows)))

# return a sentence of 6 words that ends with a GRE word
def get_gre_phrases(sent):
	sixgrams = ngrams(sent,6)
	phrases = []
	for grams in sixgrams:
		if stem.stem(grams[5]) in gre:
			phrases.append(list(grams))
	return phrases

# return a list of sentences of size 8 that ens with a GRE word
def gre_gram(sents):
	phrases = []
	for sent in sents:
		if len(sent) > 5:
			phrases2 = get_gre_phrases(sent)
			if phrases2: 
				for phrase in phrases2:
					phrases.append(phrase)
	return phrases

# Create a lexicon with the input list of sentences
def create_lexicon(file):
	lexicon = []
	for sent in file:
		for word in  sent:
			lexicon.append(stem.stem(word))
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		# if 3000 > w_counts[w] > 50:
		if w_counts[w] > 0:
			l2.append(w)
	return l2 + ['un_k']

# Transfer a word to a hot vector
def word_to_vec(word, lexicon):
	feature = np.zeros(len(lexicon))
	if word in lexicon:
		index_value = lexicon.index(word.lower())
	else:
		index_value = lexicon.index('un_k')
	
	feature[index_value] += 1
	return list(feature)

# Transfer a sent to a vector
def sent_to_vec(sent, lexicon):
	feature = []
	for word in sent:
		feature += word_to_vec(word, lexicon)
	return feature

# Transfer a list of sentences to a list of [[feature, label],...]
def text_to_num(text_list, lexicon):
	feature_set = []
	for text in text_list:
		feature = sent_to_vec(text[:-1], lexicon)
		label = np.zeros(len(gre))
		label[gre.index(stem.stem(text[-1]))] = 1
		feature_set.append([feature, label])
	return feature_set

def split(features):

	# testing_size = int(len(features)*0.1)
	testing_size = len(features) - 5000
	random.shuffle(features)
	features = np.array(features)

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

def main1():
	'''
	Only needed for the first time
	'''
	emma = gutenberg.sents() #'austen-emma.txt'
	pre = gre_gram(emma)
	with open('raw_data_set', 'wb') as fp:
		pickle.dump(pre, fp)
	lexicon = create_lexicon(pre)
	# f_set = text_to_num(pre, lexicon)
	# print(len(f_set), '\n', len(lexicon))
	with open('lexicon', 'wb') as f:
		pickle.dump(lexicon, f)

def main2():
	with open('lexicon', 'rb') as f:
		lexicon = pickle.load(f)
	with open ('raw_data_set', 'rb') as fp:
		pre = pickle.load(fp)
		f_set = text_to_num(pre, lexicon)
	return f_set, lexicon


stem = stemmer("english")
gre = gre_voc()
# with open ('raw_data_set', 'rb') as fp:
# 	pre = pickle.load(fp)
# 	lexicon = create_lexicon(pre)
# 	print(len(lexicon))
with open('lexicon', 'rb') as f:
	lexicon = pickle.load(f)
# main1()
# train_x,train_y,test_x,test_y = split(f_set)
# print(len(train_x))
# print(len(train_x[0]),'\n', train_y[0], '\n', len(train_y))
# print(len(lexicon), len(gre))


