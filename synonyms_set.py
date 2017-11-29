from PyDictionary import PyDictionary
import xlrd
import csv, re

# The dictionary to find the Synonyms
dic = PyDictionary()

# A xls reader to get all GRE vocabularies
gre_voc = xlrd.open_workbook("GRE-Vocab.xls").sheet_by_index(0)
gre = []
for row in range(gre_voc.nrows):
	gre = gre + [gre_voc.cell(row, 0).value]

# Generate a dictionary for GRE word to it's simple synonyms
# gre = ['endemic', 'venerate', 'caustic', 'viscous', 'misanthrope']
gre_dict = {word:set(dic.synonym(word)) for word in gre if dic.synonym(word)}
# print(gre_dict)
with open('./dict.csv','w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerows(gre_dict.items())


# # Build a set for possible changeable words
# changeable = set()
# synonyms = PyDictionary(gre).getSynonyms(False)
# for wordlist in synonyms:
# 	if wordlist:
# 		for word in wordlist:
# 			changeable.add(word)

# # write the changeable set to a text file
# with open('./synonym.txt','w') as file:
# 	voc = ''
# 	for word in changeable:
# 		print(word)
# 		voc = voc + word + '\n'
# 	file.write(voc)