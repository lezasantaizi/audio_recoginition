#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

import codecs
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
def process_text_1(label_file):
    with codecs.open(label_file, encoding="utf-8") as f:
        texts = f.read().split("\n"); #["我们 是 朋友","他们 不是 朋友", ...]
    del texts[-1]
    # print "texts len = %s" % len(texts)
    #print "texts[0] content: \n %s" %texts[0]

    texts = [i.split(" ") for i in texts] #把一句话拆分成词保存在texts中, [["我们","是"，"朋友"]，["他们"，"不是"，"朋友"]，...]
    all_words = [];   #这里面保存了所有的词，存在重复
    maxlen_char = 0;  #maxlen_char 是所有句子中字数最多
    labels_dict = {}  # 保存映射 ，当图片个数跟 句子个数不匹配的时候，以句子的标签为基准
    for i in np.arange(0, len(texts)):
        length = 0;
        labels_dict[texts[i][0]] = i
        for j in texts[i]:
            length += len(j);  #统计一句话中的字数，注意不是词数
        if maxlen_char <= length: maxlen_char = length;
        for j in np.arange(0, len(texts[i])):
            all_words.append(texts[i][j]);

    tok = Tokenizer(char_level=True);
    tok.fit_on_texts(all_words);
    char_index = tok.word_index;  # vocab 全部转为了字编码，char_level=True这个参数很重要，如果为False，那么继续是词的编码
    index_char = dict((char_index[i], i) for i in char_index);  #编码 --> 字
    char_vec = np.zeros((len(texts), maxlen_char), dtype=np.float32)  # 句子 --> 编码向量
    char_length = np.zeros((len(texts), 1), dtype=np.float32)   # 句子中字数

    for i in np.arange(0, len(texts)):
        j = 0;
        for i1 in texts[i][:]:
            for ele in i1:
                char_vec[i, j] = char_index[ele];
                j += 1;
        char_length[i] = j;
    return index_char,char_index,char_length,char_vec,labels_dict


def process_text_2(label_file):
    with codecs.open(label_file, encoding="utf-8") as f:
        texts = f.read().split("\n"); #["我们 是 朋友","他们 不是 朋友", ...]
    del texts[-1]

    # print "texts len = %d" %len(texts)
    #print "texts[0] content: \n %s" %texts[0]

    texts = [i.split(" ") for i in texts] #把一句话拆分成词保存在texts中, [["我们","是"，"朋友"]，["他们"，"不是"，"朋友"]，...]
    all_words = [];   #这里面保存了所有的词，存在重复
    maxlen_char = 0;  #maxlen_char 是所有句子中字数最多
    labels_dict = {}  # 保存映射 ，当图片个数跟 句子个数不匹配的时候，以句子的标签为基准
    for i in np.arange(0, len(texts)):
        length = 0;
        labels_dict[texts[i][0]] = i
        for j in texts[i][1:]:
            length += len(j);  #统计一句话中的字数，注意不是词数
        if maxlen_char <= length: maxlen_char = length;
        for j in np.arange(1, len(texts[i])):
            all_words.append(texts[i][j]);

    tok = Tokenizer(char_level=True);
    tok.fit_on_texts(all_words);
    char_index = tok.word_index;  # vocab 全部转为了字编码，char_level=True这个参数很重要，如果为False，那么继续是词的编码
    index_char = dict((char_index[i], i) for i in char_index);  #编码 --> 字
    char_vec = np.zeros((len(texts), maxlen_char), dtype=np.float32)  # 句子 --> 编码向量
    char_length = np.zeros((len(texts), 1), dtype=np.float32)   # 句子中字数

    for i in np.arange(0, len(texts)):
        j = 0;
        for i1 in texts[i][1:]:
            for ele in i1:
                char_vec[i, j] = char_index[ele];
                j += 1;
        char_length[i] = j;
    return index_char,char_index,char_length,char_vec,labels_dict

def process_vgg(img_path,char_index,char_length,char_vec,labels_dict):
    import os
    vggfeature_tensor = []
    labels_vec = []
    labels_length = []
    if img_path:
        for (dirpath, dirnames, filenames) in os.walk(img_path):
            for index, filename in enumerate(filenames):
                wav_id = os.path.basename(filename).split('_')[1]
                if labels_dict.has_key(wav_id):
                    labels_vec.append(char_vec[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                    labels_length.append(char_length[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                    fs, audio = wav.read("BAC009S0002W0122.wav")
                    inputs = mfcc(audio, samplerate=fs, numcep=num_features)
                    train_inputs = np.asarray(inputs[np.newaxis, :])
                    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
                    vggfeature_tensor.append(train_inputs.tolist())

                if index % 100 == 0: print("Completed {}".format(str(index * len(filenames) ** -1)))
    return vggfeature_tensor,labels_vec,labels_length


def decode_str(index2vocab, predict):
    str = ""
    for i in predict:
        if i < 1:
            return str
        str += index2vocab[int(i)]
    return str
# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = 100

# Hyper-parameters
num_epochs = 200
num_hidden = 128
num_layers = 1
batch_size = 1
initial_learning_rate = 0.01
momentum = 0.9

num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)

# fs, audio = wav.read("BAC009S0002W0122.wav")

# inputs = mfcc(audio, samplerate=fs,numcep=num_features)
# # Tranform in 3D array
# train_inputs = np.asarray(inputs[np.newaxis, :])
# train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)


# index_char,char_index,char_length,char_vec,labels_dict = process_text_2("aishell_transcript_v0.8.txt")
index_char,char_index,char_length,char_vec,labels_dict = process_text_2("BAC009S0002W0122.txt")
filename = "BAC009S0002W0122.wav"
labels_vec = char_vec[0]
import os
# wav_id = os.path.basename(filename).split('.')[0]
# labels_vec = char_vec[labels_dict[wav_id]]  # labels_dict[wav_id] 保存的是序号
# print(labels_vec.shape)
# labels_length.append(char_length[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
fs, audio = wav.read(filename)
inputs = mfcc(audio, samplerate=8000)
train_inputs = np.asarray(inputs[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_targets = sparse_tuple_from([labels_vec])
train_seq_len = [train_inputs.shape[1]]
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

target_str = decode_str(index_char, labels_vec)
print(target_str)
graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)
    #dense = tf.sparse_tensor_to_dense(targets)
    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    # # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # f1_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
    # b1_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

    f1_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    b1_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell,b1_cell,inputs,seq_len,dtype=tf.float32)
    # cell = tf.contrib.rnn.GRUCell(num_hidden)
    # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
    # outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    merge = tf.concat(outputs,axis = 2)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(merge, [-1, num_hidden*2])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden * 2,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(initial_learning_rate,momentum).minimize(cost)#效果更快
    optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    dense = tf.sparse_to_dense(decoded[0].indices,decoded[0].dense_shape,decoded[0].values)
    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}
            #run_sense = session.run([dense],feed)
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
    # Decoding
    d = session.run(dense, feed_dict=feed)





    str = decode_str(index_char, d[0])
    print(str)



