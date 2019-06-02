from nltk.corpus import PlaintextCorpusReader
corpus_root = './data'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
print(wordlists.fileids())
print(wordlists.words('firefox.txt'))