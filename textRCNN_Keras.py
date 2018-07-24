# -*- coding: utf-8 -*-
import json
import pickle
import gensim
import os
from gensim.models import word2vec
from keras import backend
from keras.utils import np_utils
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping
import logging
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

w2vpath = "./predictor/model/word2vec"  #w2v模型地址
embedding_dims = 100  # 词向量长度
logpath='./model/mylog.txt' #日志记录地址
modelpath='./model/' #模型保存目录
#模型训练参数
batch_size = 32
epochs = 15
#TextRCNNmodel参数
hidden_dim_1 = 200
hidden_dim_2 = 100

def init():
	f = open('./datanew/law.txt', 'r', encoding='utf8')
	law = {}             # law{0: '184', 1: '336', 2: '314', ....}
	lawname = {}         # lawname{184:0,336:2,...}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	# print(lawname)
	f.close()

	f = open('./datanew/accu.txt', 'r', encoding='utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()  # {{0: '妨害公务', 1: '寻衅滋事', 2: '盗窃、侮辱尸体'
		accu[line.strip()] = len(accu)      # {'寻衅滋事': 1, '绑架': 8, ',...}
		line = f.readline()
	# print(accu)
	f.close()

	return law, accu, lawname, accuname


def get_name(index, kind):
	global lawname
	global accuname
	if kind == 'law':
		return lawname[index]

	if kind == 'accu':
		return accuname[index]


def get_time(time):
	# 将刑期用分类模型来做
	v = int(time['imprisonment'])

	if time['death_penalty']:
		return 0
	if time['life_imprisonment']:
		return 1
	elif v > 10 * 12:
		return 2
	elif v > 7 * 12:
		return 3
	elif v > 5 * 12:
		return 4
	elif v > 3 * 12:
		return 5
	elif v > 2 * 12:
		return 6
	elif v > 1 * 12:
		return 7
	else:
		return 8


def get_label(d, kind):
	global law
	global accu

	# 做单标签

	if kind == 'law':
		# 返回多个类的第一个
		return law[str(d['meta']['relevant_articles'][0])]
	if kind == 'accu':
		return accu[d['meta']['accusation'][0]]

	if kind == 'time':
		return get_time(d['meta']['term_of_imprisonment'])


def read_data():
	print('reading train data...')

	train_data = []
	with open('./cuttext_all_large.txt',encoding='UTF-8') as f:
		train_data = f.read().splitlines()
	print(len(train_data))        # 154592

	path = './datanew/cail2018_big.json'
	fin = open(path, 'r', encoding='utf8')

	accu_label = []
	law_label = []
	time_label = []

	line = fin.readline()
	while line:
		d = json.loads(line)
		accu_label.append(get_label(d, 'accu'))
		law_label.append(get_label(d, 'law'))
		time_label.append(get_label(d, 'time'))
		line = fin.readline()
	fin.close()

	print('reading train data over.')
	return train_data,accu_label, law_label, time_label

def slice_data(slice_size=None):
	if slice_size is None:
		alltext, accu_label, law_label, time_label = read_data()
	else:
		alltext, accu_label, law_label, time_label = read_data()

		# randnum = random.randint(0,len(alltext))
		# random.seed(randnum)
		# random.shuffle(alltext)
		# random.seed(randnum)
		# random.shuffle(law_label)
		# random.seed(randnum)
		# random.shuffle(accu_label)
		# random.seed(randnum)
		# random.shuffle(time_label)

		alltext = alltext[:slice_size]
		law_label = law_label[:slice_size]
		accu_label = accu_label[:slice_size]
		time_label = time_label[:slice_size]

	return alltext, law_label, accu_label, time_label

def data_process():
    print('Pad sequences...')
    train_data, accu_label, law_label, time_label = slice_data()
    # 转换成词袋序列
    maxlen = 500
    # 词袋模型的最大特征束
    # max_features = 20000

    # 设置分词最大个数 即词袋的单词个数
    with open('./predictor/model/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    # tokenizer = Tokenizer(num_words=max_features, lower=True)  # 建立一个max_features个词的字典
    tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
    global word_index
    word_index = tokenizer.word_index  # 长度为508242
    sequences = tokenizer.texts_to_sequences(
        train_data)  # 对每个词编码之后，每个文本中的每个词就可以用对应的编码表示，即每条文本已经转变成一个向量了 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
    x = sequence.pad_sequences(sequences, maxlen,dtype='int16')  # 将每条文本的长度设置一个固定值。
    del tokenizer, sequences
    y = np_utils.to_categorical(accu_label)  # 多分类时，此方法将1，2，3，4，....这样的分类转化成one-hot 向量的形式，最终使用softmax做为输出
    print('x shape and y shape are:',x.shape, y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    x_train = x[indices][:int(lenofdata * 0.8)]
    y_train = y[indices][:int(lenofdata * 0.8)]
    x_test = x[indices][int(lenofdata * 0.8):]
    y_test = y[indices][int(lenofdata * 0.8):]
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, y_train, x_test, y_test

def train(x_train, y_train, x_test, y_test,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,hidden_dim_1,hidden_dim_2):
    print(modelname + 'Build model...')
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    model = gensim.models.Word2Vec.load('./predictor/model/word2vec')
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    max_token = len(model.wv.vocab.items())
    print('Found %s word vectors.' % len(model.wv.vocab.items()))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedding_matrix[i + 1] = vocab_list[i][1]

    embedder = Embedding(max_token + 1, embedding_dims, weights=[embedding_matrix], trainable=False) #input_length=maxlen
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)

    # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
    forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)  # See equation (1).
    backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2).
    together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

    semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)  # See equation (4).

    # Keras provides its own max-pooling layers, but they cannot handle variable length input
    # (as far as I can tell). As a result, I define my own max-pooling layer here.
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (5).

    output = Dense(1, input_dim=hidden_dim_2, activation="softmax")(pool_rnn)  # See equations (6) and (7).NUM_CLASSES=1

    model = Model(inputs=[document, left_context, right_context], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    ##生成左右上下文
    print('Build left and right data')
    doc_x_train = np.array(x_train)
    # We shift the document to the right to obtain the left-side contexts.
    left_x_train = np.array([[max_token] + t_one[:-1].tolist() for t_one in x_train])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_train = np.array([t_one[1:].tolist() + [max_token] for t_one in x_train])

    doc_x_test = np.array(x_test)
    # We shift the document to the right to obtain the left-side contexts.
    left_x_test = np.array([[max_token] + t_one[:-1].tolist() for t_one in x_test])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_test = np.array([t_one[1:].tolist() + [max_token] for t_one in x_test])

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    # history = model.fit([doc_x_train, left_x_train, right_x_train], y_train, epochs = 1)
    # loss = history.history["loss"][0]
    hist = model.fit([doc_x_train, left_x_train, right_x_train], y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=[[doc_x_test, left_x_test, right_x_test], y_test], callbacks=[early_stopping])

    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')

# Data Preparation
law, accu, lawname, accuname = init()
print("Load data...")
x_train, y_train, x_test, y_test = data_process()

train(x_train, y_train, x_test, y_test,
      embedding_dims,batch_size,epochs,logpath,modelpath,
      "TextRCNN",hidden_dim_1,hidden_dim_2)

