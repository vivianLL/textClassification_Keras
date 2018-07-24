import json
import joblib
import multiprocessing
import pickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from keras.preprocessing import sequence,text
from keras.optimizers import RMSprop
from keras.utils import np_utils,plot_model
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Input,Convolution1D,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten,concatenate,Embedding,GRU,Lambda, LSTM, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from datetime import datetime
import gensim
import numpy as np
import os
import thulac
from gensim.models import word2vec
import codecs
import tensorflow as tf
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def init():
	f = open('law.txt', 'r', encoding='utf8')
	law = {}             # law{0: '184', 1: '336', 2: '314', ....}
	lawname = {}         # lawname{184:0,336:2,...}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	# print(lawname)
	f.close()

	f = open('accu.txt', 'r', encoding='utf8')
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


def read_train_data(path):
	print('reading train data...')
	fin = open(path, 'r', encoding='utf8')

	# alltext = []

	accu_label = []
	law_label = []
	time_label = []

	line = fin.readline()
	while line:
		d = json.loads(line)
		# alltext.append(d['fact'])
		accu_label.append(get_label(d, 'accu'))
		law_label.append(get_label(d, 'law'))
		time_label.append(get_label(d, 'time'))
		line = fin.readline()
	fin.close()

	train_data = []
	with open('./cuttext_all_large.txt') as f:
		train_data = f.read().splitlines()
	print(len(train_data))  # 154592

	return train_data,accu_label, law_label, time_label


def slice_data(slice_size=None):
	if slice_size is None:
		alltext,accu_label, law_label, time_label = read_train_data(path='./datanew/cail2018_big.json')
	else:
		alltext,accu_label, law_label, time_label = read_train_data(path='./datanew/cail2018_big.json')

		alltext = alltext[:slice_size]
		law_label = law_label[:slice_size]
		accu_label = accu_label[:slice_size]
		time_label = time_label[:slice_size]

	return alltext,law_label, accu_label, time_label

def cut_text(alltext):
	print('cut text...')
	count = 0
	cut = thulac.thulac(seg_only=True)
	train_text = []
	for text in alltext:
		count += 1
		if count % 2000 == 0:
			print(count)
		train_text.append(cut.cut(text, text=True)) #分词结果以空格间隔，每个fact一个字符串
	print(len(train_text))

	print(train_text)
	fileObject = codecs.open("./predictor/cuttext_5000.txt", "w", "utf-8")  #必须指定utf-8否则word2vec报错
	for ip in train_text:
		fileObject.write(ip)
		fileObject.write('\n')
	fileObject.close()
	print('cut text over')
	return train_text

def word2vec_train():

	sentences = word2vec.Text8Corpus("cuttext.txt")
	model = word2vec.Word2Vec(sentences)
	model.save('./predictor/model/word2vec.m')
	print(model.similarity('被害人','发生'))
	# print(model.most_similar("被告人"))
	print('finished and saved!')
	return model


	# 构造神经网络
def baseline_model(y,max_features,embedding_dims,filters):
	kernel_size = 3

	model = Sequential()
	model.add(Embedding(max_features, embedding_dims))        # 使用Embedding层将每个词编码转换为词向量
	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	# 池化
	model.add(GlobalMaxPooling1D())

	model.add(Dense(y.shape[1], activation='softmax')) #第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
	model.add(Dropout(0.2))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.summary()

	return model

def test_cnn(y,maxlen,max_features,embedding_dims,filters = 250):
	#Inputs
	seq = Input(shape=[maxlen],name='x_seq')

	#Embedding layers
	emb = Embedding(max_features,embedding_dims)(seq)

	# conv layers
	convs = []
	filter_sizes = [2,3,4,5]
	for fsz in filter_sizes:
		conv1 = Conv1D(filters,kernel_size=fsz,activation='tanh')(emb)
		pool1 = MaxPooling1D(maxlen-fsz+1)(conv1)
		pool1 = Flatten()(pool1)
		convs.append(pool1)
	merge = concatenate(convs,axis=1)

	out = Dropout(0.5)(merge)
	output = Dense(32,activation='relu')(out)

	output = Dense(units=y.shape[1],activation='sigmoid')(output)

	model = Model([seq],output)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

def cnn_w2v(y,max_features,embedding_dims,filters,maxlen):
	# CNN参数
	kernel_size = 3

	# model = gensim.models.Word2Vec.load('./predictor/model/word2vec')
	# # 得到一份字典(embeddings_index)，每个词对应属于自己的300维向量
	# embeddings_index = {}
	# word_vectors = model.wv
	# for word, vocab_obj in model.wv.vocab.items():
	# 	if int(vocab_obj.index) < max_features:
	# 		embeddings_index[word] = word_vectors[word]
	# del model, word_vectors   # 删掉gensim模型释放内存
	# print('Found %s word vectors.' % len(embeddings_index))
    #
    #
	# embeddings_matrix = np.zeros((len(word_index)+1,embedding_dims))  # 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
	# for word,i in word_index.items(): # 从索引为1的词语开始，用词向量填充矩阵
	# 	if i>=max_features:
	# 		continue
	# 	embedding_vector = embeddings_index.get(word)
	# 	if embedding_vector is not None:
	# 		# 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
	# 		embeddings_matrix[i] = embedding_vector # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充

	model = gensim.models.Word2Vec.load('./predictor/model/word2vec')
	word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	print('Found %s word vectors.' % len(model.wv.vocab.items()))
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i + 1
		embeddings_matrix[i + 1] = vocab_list[i][1]

	model = Sequential()
	# 使用Embedding层将每个词编码转换为词向量
	model.add(Embedding(len(embeddings_matrix),       #表示文本数据中词汇的取值可能数,从语料库之中保留多少个单词。 因为Keras需要预留一个全零层， 所以+1
								embedding_dims,       # 嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小
								weights=[embeddings_matrix], #构建一个[num_words, EMBEDDING_DIM]的矩阵,然后遍历word_index，将word在W2V模型之中对应vector复制过来。换个方式说：embedding_matrix 是原始W2V的子集，排列顺序按照Tokenizer在fit之后的词顺序。作为权重喂给Embedding Layer
								input_length=maxlen,     # 输入序列的长度，也就是一次输入带有的词汇个数
								trainable=False        # 我们设置 trainable = False，代表词向量不作为参数进行更新
						))
	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	# 池化
	model.add(GlobalMaxPooling1D())

	model.add(Dense(y.shape[1], activation='softmax')) #第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
	model.add(Dropout(0.2))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.summary()

	return model

# def runcnn(train_data,label,label_name):
def runcnn(x,label, label_name):
    y = np_utils.to_categorical(label) #多分类时，此方法将1，2，3，4，....这样的分类转化成one-hot 向量的形式，最终使用softmax做为输出
    print(x.shape,y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    x_train = x[indices][:int(lenofdata*0.8)]
    y_train = y[indices][:int(lenofdata*0.8)]
    x_test = x[indices][int(lenofdata*0.8):]
    y_test = y[indices][int(lenofdata*0.8):]
    max_features=20000   # 词汇表大小
    maxlen = 1000        # 序列最大长度
    embedding_dims=100    # 词向量维度
    filters = 250

	# model = test_cnn(y,maxlen,max_features,embedding_dims,filters)
    model = baseline_model(y,max_features,embedding_dims,filters)
    # model = cnn_w2v(y,max_features,embedding_dims,filters,maxlen)
    earlyStopping = keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=0,  # 当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
			verbose=0,
			mode='auto')

    print("training model")
    model.fit(x_train,y_train,validation_split=0.2,batch_size=32,epochs=10,verbose=2,callbacks=[earlyStopping],shuffle=True,class_weight='auto') #自动设置class weight让每类的sample对损失的贡献相等


    print("pridicting...")
    scores = model.evaluate(x_test,y_test)
    print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))

    print("saving %s_model" % label_name)
    # model.save('./predictor/model/%s_cnn.h5' % label_name)
    model.save('./predictor/model/%s_textcnn.h5' % label_name)
    # model.save('./predictor/model/%s_w2vcnn.h5' % label_name)

def cnn_triple(train_data,accusation, law, time):
	print('train_cnn')



	# 词袋模型的最大特征束
	max_features = 20000
	maxlen = 1000

	# 设置分词最大个数 即词袋的单词个数
	tokenizer = Tokenizer(num_words=max_features, lower=True)  # 建立一个max_features个词的字典
	tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
	# with open('./predictor/model/tokenizer.pickle', 'wb') as handle:
	# 	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# print("tokenizer has been saved.")
	# print(len(tokenizer.word_index))    # 508241个词

	sequences = tokenizer.texts_to_sequences(
		train_data)  # 对每个词编码之后，每个文本中的每个词就可以用对应的编码表示，即每条文本已经转变成一个向量了 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)

	x = sequence.pad_sequences(sequences, maxlen)  # 将每条文本的长度设置一个固定值。

	runcnn(x,accusation, "accu")
	runcnn(x,law, "law")
	runcnn(x,time, "time")

if __name__ == "__main__":
	start_time = datetime.now()
	law, accu, lawname, accuname = init()

	# datanumber = 5000   # 仅取数据的前50000个
	train_data,law_label, accu_label, time_label = slice_data(500000)

	# train_data = cut_text(alltext)

	cnn_triple(train_data,accu_label, law_label, time_label)
	print(datetime.now()-start_time, 'seconds')