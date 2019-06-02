from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1])


#创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
#将数据集分为训练集和验证集
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
#加载预先训练好的词嵌入向量
embeddings_index = {}
#下载网址：https://fasttext.cc/docs/en/english-vectors.html
for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
     values = line.split()
     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')


     #创建一个分词器
     token = text.Tokenizer()
     token.fit_on_texts(trainDF['text'])
     word_index = token.word_index


     #将文本转换为分词序列，并填充它们保证得到相同长度的向量
     train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
     valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)


      #创建分词嵌入映射
     embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
     for word, i in word_index.items():
           embedding_vector = embeddings_index.get(word)
     if embedding_vector is not None:
           embedding_matrix[i] = embedding_vector
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    else:

     return metrics.accuracy_score(predictions, valid_y)
# label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
print(train_y)

def create_rnn_lstm():
     # Add an Input Layer
     input_layer = layers.Input((70, ))

     # Add the word embedding Layer
     embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
     embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

     # Add the LSTM Layer
     lstm_layer = layers.LSTM(100)(embedding_layer)

     # Add the output Layers
     output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
     output_layer1 = layers.Dropout(0.25)(output_layer1)
     output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

     # Compile the model
     model = models.Model(inputs=input_layer, outputs=output_layer2)
     model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
     return model

classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-LSTM, Word Embeddings", accuracy)