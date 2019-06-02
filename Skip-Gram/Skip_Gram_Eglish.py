import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter
#https://www.jianshu.com/p/a2e6a487b385

with open('./data/text8.txt') as f:
    text = f.read()


#定义函数来完成数据的预处理
def preprocess(text, freq=5):
    '''
    对文本进行预处理
    :param text: 文本数据
    :param freq: 词频阈值
    '''
    #对文本中的符号进行替换，下面都是用英文翻译来替代符号
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    #删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq] #这句话很好，可以模仿

    return trimmed_words


#清洗文本并分词
words = preprocess(text)
# print(words[: 200])

#构建映射表
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

print('total words: {}'.format(len(words))) #还有这种输出方法，学习一下
print('unique words: {}'.format(len(set(words))))

#对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]


#--------------------------------------------------------------------采样

t = 1e-5  #t值
threshold = 0.8  #删除概率阈值

#统计单词出现频数
int_word_counts = Counter(int_words)
total_count = len(int_words)
#计算单词频率
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
#计算单词被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
#对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]
print(len(train_words))
#-----------------------------------------------------------------------采样


#获得input word的上下文单词列表
def get_targets(words, idx, window_size = 5):
    '''
    获得input word的上下文单词列表
    :param words: 单词列表
    :param idx: input word 的索引号
    :param window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)  #从1到 window_size+1 之间的数，包括1，不包括最后一个
    #这里要考虑input word前面单词不够的情况，但是为什么没有写后面单词不够的情况呢，
    #因为python里面的list取分片越界时会自动只取到结尾的
    start_point = idx - target_window if (idx - target_window) > 0 else 0  #虽说不能写三元表达式，但这个挺不错的
    end_point = idx + target_window
    #output words(即窗口中的上下文单词，不包含目标词，只是它的上下文的词)
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])

    return list(targets)


#构造一个获取batch的生成器
def get_batches(words, batch_size, window_size = 5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(words) // batch_size

    #仅取full batches
    words = words[: n_batches * batch_size]

    for idx in range(0, len(words), batch_size):  #range(start, stop[, step])
        x, y = [], []
        batch = words[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size) #从一个batch的第0位开始，一直往下滑，直到最后一个
            #由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


#构建网络，该部分主要包括：输入层，Embedding，Negative Sampling
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')  #一般是[batch_size, num_true]，num_true一般为1

vocab_size = len(int_to_vocab)
embedding_size = 200 #嵌入维度

with train_graph.as_default():
    #嵌入层权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1)) #tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))返回4*4的矩阵
                                                                                    # ，产生于low和high之间，产生的值是均匀分布的。
    embed = tf.nn.embedding_lookup(embedding, inputs) #[None, embedding_size]，一般是[batch_size, embedding_size]


#-----------------------------------------------------------负采样（Negative Sampling）
n_sampled = 100

with train_graph.as_default():
    sotfmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1)) #[vocab_size, dim], dim就是embedding_size
    sotfmax_b = tf.Variable(tf.zeros(vocab_size)) #[vocab_size]

    #计算negative sampling下的损失
    #tf.nn.sampled_softmax_loss()进行了negative sampling，它主要用在分类的类别较大的情况
    loss = tf.nn.sampled_softmax_loss(sotfmax_w, sotfmax_b, labels, embed, n_sampled, vocab_size)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
#-----------------------------------------------------------负采样


#为了直观看到训练结果，我们将查看训练出的相近语义的词
with train_graph.as_default():
    #随机挑选一些单词
    valid_size = 16
    valid_window = 100
    #从不同位置各选8个词
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))  #random.sample(seq, n) 从序列seq中选择n个随机且独立的元素
    #np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])，结果是array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_size = len(valid_examples)
    #验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    #计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.squeeze(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    #查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    #计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


#进行训练
epochs = 10 #迭代次数
batch_size = 1000 #batch大小
window_size = 10 #窗口大小

with train_graph.as_default():
    saver = tf.train.Saver() #文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()

        for x, y in batches:
            feed = {
                inputs : x,
                labels : np.array(y)[:, None]
            }
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print('Epoch {}/{}'.format(e, epochs),
                      'Iteration: {}'.format(iteration),
                      'Avg. Training loss: {:.4f}'.format(loss / 100),  #这是一种数字格式化的方法，{:.4f}表示保留小数点后四位
                      '{:.4f} sec / batch'.format((end-start) / 100))
                loss = 0
                start = time.time()

            #计算相似的词
            if iteration % 1000 == 0:
                #计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 #取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s%s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, 'model/text8.ckpt')
    embed_mat = sess.run(normalized_embedding)


