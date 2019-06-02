from nltk.corpus import stopwords
import nltk
print(stopwords.words('english'))
def content_fraction(text):
    stopwords=nltk.corpus.stopwords.words('english')
    content= [w for w in text if w.lower()not in stopwords]
    return len(content) /len(text)
print(content_fraction(nltk.corpus.reuters.words()))