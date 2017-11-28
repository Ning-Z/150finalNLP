from PyDictionary import PyDictionary
from nltk.stem.snowball import SnowballStemmer as stemmer
import xlrd

# The dictionary to find the Synonyms
dic = PyDictionary()
# The Stemmer to get the right tense of a word
stem = stemmer("english")

# A xls reader to get all GRE vocabularies
gre_voc = xlrd.open_workbook("GRE-Vocab.xls").sheet_by_index(0)
gre = []
for row in range(gre_voc.nrows):
	gre = gre + [gre_voc.cell(row, 0).value]
gre = ['endemic', 'venerate', 'caustic', 'viscous', 'misanthrope']


# Build a set for possible changeable words
changeable = set()
synonyms = PyDictionary(gre).getSynonyms(False)
print(synonyms)
for wl in synonyms:
	for w in wl:
		changeable.add(w)
print(changeable)

''' 
For all the words from the text that is in changeable 
set will be changes with GRE word
'''
def replaceable(file):
	set()

def replace():