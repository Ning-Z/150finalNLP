# 150finalNLP
This a text "complifier", which will replace simple words in a text with more complicated GRE words.

Two levels of replacements are implied here: word level and sentence level.

## Word level
simply check the synonyms with word process APIs and decide which word is repalceable and how to replace
## Sentence level
Build up a LSTM model with Tensorflow and trained with the nltk corpus to choose a GRE word for the replaceable word.
