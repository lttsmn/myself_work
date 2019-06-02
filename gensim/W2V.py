from gensim.models import word2vec
import logging
raw_sentence=["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]
sentence=[s.split() for s in raw_sentence]
print(sentence)
model=word2vec.Word2Vec(sentence,min_count=1)
#min_count:过滤掉低频词，词频小于min_count的单词舍弃
#size:主要用于设置神经网络的参数。Wprd2Vec中默认值是设置为100层。更大的层数意味着更多的输入，不过也会相应的提高准确率
print(model.similarity("dogs","you"))