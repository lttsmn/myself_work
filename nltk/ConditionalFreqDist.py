from nltk.corpus import brown
import nltk
'''cfd = nltk.ConditionalFreqDist(
     (genre, word)
     for genre in brown.categories()
     for word in brown.words(categories=genre))'''
from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
     (target, fileid[:4])
     for fileid in inaugural.fileids()
     for w in inaugural.words(fileid)
     for target in ['america', 'citizen']
     if w.lower().startswith(target))
print(sorted(list(cfd['america'])))
#cfd.plot()