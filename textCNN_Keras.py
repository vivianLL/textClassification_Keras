import numpy as np
import os
import json
import pickle
import gensim
import random
from gensim.models import word2vec
from os.path import join, exists, split
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
embedding_dim = 100
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 32
num_epochs = 10

# Prepossessing parameters
sequence_length = 1500   #400
max_words = 20000   #5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------
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

law, accu, lawname, accuname = init()

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

def slice_data(slice_size=None):
	if slice_size is None:
		alltext, accu_label, law_label, time_label = read_data()
	else:
		alltext, accu_label, law_label, time_label = read_data()

		randnum = random.randint(0,len(alltext))
		random.seed(randnum)
		random.shuffle(alltext)
		random.seed(randnum)
		random.shuffle(law_label)
		random.seed(randnum)
		random.shuffle(accu_label)
		random.seed(randnum)
		random.shuffle(time_label)

		alltext = alltext[:slice_size]
		law_label = law_label[:slice_size]
		accu_label = accu_label[:slice_size]
		time_label = time_label[:slice_size]

	return alltext, law_label, accu_label, time_label

def read_data():
	print('reading train data...')

	train_data = []
	with open('./cuttext_all_large.txt') as f:
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


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """

    model_name = './predictor/model/word2vec'
    if exists(model_name):
        # embedding_model = word2vec.Word2Vec.load(model_name)
        embedding_model = gensim.models.Word2Vec.load('./predictor/model/word2vec')
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in embedding_model.wv.vocab.items()}
    return embedding_weights

def data_process():
    train_data, accu_label, law_label, time_label = slice_data(1000000)
    # 转换成词袋序列
    maxlen = 500
    # 词袋模型的最大特征束
    max_features = 20000

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
    print(x.shape, y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    x_train = x[indices][:int(lenofdata * 0.8)]
    y_train = y[indices][:int(lenofdata * 0.8)]
    x_test = x[indices][int(lenofdata * 0.8):]
    y_test = y[indices][int(lenofdata * 0.8):]

    model = word2vec.Word2Vec.load('./predictor/model/word2vec')
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    # vocabulary_inv = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    vocabulary_inv = dict((k, model.wv[k]) for k, v in model.wv.vocab.items())

    return x,y,x_train, y_train, x_test, y_test, vocabulary_inv


# Data Preparation
print("Load data...")
x,y,x_train, y_train, x_test, y_test, vocabulary_inv = data_process()

w = train_word2vec(x, vocabulary_inv)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(y.shape[1], activation="softmax")(z)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Train the model
print("training...")
model.fit(x_train, y_train,validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=2,class_weight='auto')

print("pridicting...")
scores = model.evaluate(x_test,y_test)
print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))

print("saving accu_textcnnmodel")
model.save('./predictor/model/accu_textcnn.h5')