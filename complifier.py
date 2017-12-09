from PyDictionary import PyDictionary
from nltk.stem.snowball import SnowballStemmer as stemmer
from nltk import sent_tokenize, word_tokenize
import xlrd, re, csv
import nltk
nltk.download('punkt')

def changeable():
	# read in the changeable word vocabulary
	return set(line.strip('\n') for line in open('synonym.txt'))

''' 
For all the words from the text that is in changeable 
set will be changes with GRE word
We will return the word to be replaced and the original text
'''
def replaceable(change,input):
	# with open('./PeterPan.txt' , "r") as file:
	# 	head = file.readlines()[:80]
	head = input
	text = ''
	for lines in head:
		text += lines.replace('\n',' ')		
	#wl = re.findall('\w+', text)
	wl = word_tokenize(text)
	return set(w.lower() for w in wl if stem.stem(w.lower()) in change), wl

# Calculate the replacement ratio, to_be_repalced/total_voc
def replace_rotio(replace_voc ,total_voc):
	r_voc = set(stem.stem(w) for w in replace_voc)
	total_voc = set(stem.stem(w) for w in total_voc)
	return float(len(r_voc))/float(len(total_voc))

# replace the word with a GRE word, using a gre_synonyms dictionary
def sigle_sub(word, gre_syndict):
	# search through the gre dict and check if this word is in the synonyms list
	# if we find the word, then keep it
	sub = []
	for gre in gre_syndict:
		if stem.stem(word) in gre_syndict[gre]:
			sub = sub + [gre]
	return sub

# replace the word with synonyms dictionary (choose one with above)
def sigle_sub_stem(word, gre_set):
	# get the synonyms for the word
	# stem the synonyms and check if it's in GRE voc, if it is keep it
	synonyms = dictionary.synonym(stem.stem(word))
	return [w for w in synonyms if w in gre_set]

# According to the replaceable set, substitute the replaceable one by one 
def substitute(replaceable, text, gre_set):
	new_text=[]
	for word in text:
		if word in replaceable:
			new_text.append({word: sigle_sub(word, gre_dict)})
			# new_text.append({word: sigle_sub_stem(word, gre)})
		else:
			new_text.append(word)
	return new_text

# this function is for creating the tooltip
def single_tooltip_map(word):
	greString = ""
	grelist = sigle_sub(word, gre_dict)
	for greword in grelist:
		greString = greString + greword + "<br>"
	tooltipMap = '<div class="mytooltip">'+word+'<span class="mytooltiptext">'+greString+'</span></div>'
	return tooltipMap

def tooltip(replaceable, text, gre_set):
	tooltip_map = ""
	for word in text:
		if word in replaceable:
			tooltip_map += single_tooltip_map(word)
		else:
			tooltip_map = tooltip_map + word + " "
	return tooltip_map

# Read in the original GRE vocabulary as a set
def gre_voc():
	# A xls reader to get all GRE vocabularies
	gre_voc = xlrd.open_workbook("GRE-Vocab.xls").sheet_by_index(1)
	return set(gre_voc.cell(row, 0).value for row in range(gre_voc.nrows))

# Read in the dictionary for gre:list[synonyms] (convenient for back track)
def gre_synonyms():
	with open('./dict.csv', 'r') as file:
		reader = csv.reader(file)
		dic = {}
		for row in reader:
			# print(row)
			dic[row[0]] = set(re.findall('\w+', row[1]))
	return dic

# The Stemmer to get the right tense of a word
stem = stemmer("english")
# gre = ['endemic', 'venerate', 'caustic', 'viscous', 'misanthrope']
dictionary = PyDictionary()
gre_set = gre_voc()
gre_dict = gre_synonyms()
change = changeable()
# replace, text = replaceable(change)
# print(replace)
# print(len(replace))
# print(replace_rotio(replace, text))
# new = substitute(replace, text, gre)
# print(new)

def complify(text):
	replace, text = replaceable(change,text)
	new = substitute(replace, text, gre_set)
	return new

def get_ReplaceRatio(text):
	replace, text = replaceable(change,text)
	return replace_rotio(replace,text)
# def replace():

# learning replace need tokenizer as below 
# sentences = sent_tokenize(text)
		# print(sentences)
		# for sent in sentences:
		# 	print(re.findall('\w+', sent))

def tooltip_output(text):
	if text:
		replace, text = replaceable(change,text)
		return tooltip(replace, text, gre_set)
	else: return ""
