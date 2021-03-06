#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import wraps

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range
import os
import codecs
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from python_speech_features import mfcc

def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__+'...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__+' in '+ str(end-start)+' s'))
        return result
    return wrapper

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return [indices, values, shape]
@describe
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
@describe
def process_vgg(img_path,char_index,char_length,char_vec,labels_dict,compute_sample_num):

    vggfeature_tensor = []
    labels_vec = []
    labels_length = []
    seq_length = []
    sample_index = 0
    if img_path:
        for (dirpath_1,dirnames_1,_) in os.walk(img_path):
            for one_dir in sorted(dirnames_1):
                for (dirpath_2, _, filenames_2) in os.walk(os.path.join(dirpath_1,one_dir)):
                    for index, filename in enumerate(sorted(filenames_2)):
                        if filename.endswith("wav"):
                            wav_id = os.path.basename(filename).split('.')[0]
                            if labels_dict.has_key(wav_id):
                                labels_vec.append(char_vec[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                                labels_length.append(char_length[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                                fs, audio = wav.read(os.path.join(dirpath_2,filename))
                                inputs = mfcc(audio, samplerate=fs, numcep=num_features)
                                seq_length.append(inputs.shape[0])
                                train_inputs = np.asarray(inputs[np.newaxis, :])
                                train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
                                vggfeature_tensor.append(train_inputs)
                                sample_index = sample_index + 1
                                if sample_index >= compute_sample_num:
                                    return vggfeature_tensor, np.array(labels_vec), np.array(labels_length), np.array(
                                    seq_length)

                        if sample_index % 100 == 0: print("Completed {}".format(str(sample_index * compute_sample_num ** -1)))
    return vggfeature_tensor,np.array(labels_vec),np.array(labels_length),np.array(seq_length)

def decode_str(index2vocab, predict):
    str = ""
    for i in predict:
        if i < 1:
            return str
        str += index2vocab[int(i)]
    return str



@describe
def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num

class Model():
    def get_layer_shape(self, layer):
        thisshape = tf.Tensor.get_shape(layer)
        ts = [thisshape[i].value for i in range(len(thisshape))]
        return ts
    def __init__(self,num_features,num_layers,num_hidden,num_classes,batch_size,max_seq_len):
        self.inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len, num_features])
        self.targets = tf.sparse_placeholder(tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])

        # # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # f1_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
        # b1_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
        inputs = self.inputs
        for i in range(num_layers):
            f1_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            b1_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            scope = 'bilstm_' + str(i + 1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, inputs, self.seq_len, dtype=tf.float32,scope= scope)
            merge = tf.concat(outputs, axis=2)
            shape = merge.get_shape().as_list()
            output_fb = tf.reshape(merge, [shape[0], shape[1], 2, int(shape[2] / 2)])
            inputs = tf.reduce_sum(output_fb, 2)
            #inputs = tf.contrib.layers.dropout(merge, keep_prob=keep_prob, is_training=is_training)

        shape = tf.shape(self.inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        outputs = tf.reshape(inputs, [-1, self.get_layer_shape(inputs)[-1]])
        W = tf.Variable(tf.truncated_normal([self.get_layer_shape(inputs)[-1],num_classes],stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(self.targets, logits, self.seq_len)
        self.cost = tf.reduce_mean(loss)

        # optimizer = tf.train.MomentumOptimizer(initial_learning_rate,momentum).minimize(cost)#效果更快
        self.optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.cost)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.seq_len)
        self.dense = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.targets))
        self.var_op = tf.global_variables()
        self.var_trainable_op = tf.trainable_variables()

#生成训练数据

num_features = 20
num_epochs = 300
num_hidden = 256
num_layers = 2
batch_size = 50
initial_learning_rate = 0.01
compute_sample_num = 500
keep = False

index_char,char_index,char_length,char_vec,labels_dict = process_text_2("aishell_small.txt")
train_inputs,labels_vec,labels_length,train_seq_len = \
    process_vgg("/mnt/steven/data/data_aishell/wav/train",char_index,char_length,char_vec,labels_dict,compute_sample_num)

new_train = np.zeros([len(train_inputs),np.max(train_seq_len),train_inputs[0].shape[2]])
for i in range(len(train_inputs)):
    new_train[i,:train_seq_len[i]] = train_inputs[i][0,:]
train_inputs = new_train
print( "label_vec_shape = %s, vocab len = %d" %(labels_vec.shape,len(index_char)))

#根据训练数据 设置参数
num_classes = len(index_char) + 2
num_examples = train_inputs.shape[0]

num_batches_per_epoch = int(num_examples/batch_size)


train_targets = []
for index in range(labels_vec.shape[0] // batch_size):
    train_targets.append(sparse_tuple_from(labels_vec[index * batch_size : (index+1)*batch_size,:]))

#设置验证集合
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

for i in labels_vec:
    target_str = decode_str(index_char, i)
    print(target_str)

#设置模型
asr_model = Model(num_features,num_layers,num_hidden,num_classes,batch_size,np.max(train_seq_len))
num_params = count_params(asr_model, mode='trainable')
print("num_params = %d " % num_params)
saver = tf.train.Saver()
save_path = os.path.join(os.getcwd(),"checkpoint_dir")
if os.path.isdir(save_path):
    None
else:
    os.mkdir(save_path)
with tf.Session() as session:
    if keep:
        saver.restore(session, tf.train.latest_checkpoint(save_path))
        print('Model restored from:' + save_path)
    else:
        print('Initializing')
        tf.global_variables_initializer().run()


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

            feed = {asr_model.inputs: train_inputs[batch * batch_size : (batch + 1) * batch_size],
                    asr_model.targets: train_targets[batch],
                    asr_model.seq_len: train_seq_len[batch * batch_size : (batch + 1) * batch_size]}
            #run_sense = session.run([dense],feed)
            batch_cost, _ = session.run([asr_model.cost, asr_model.optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(asr_model.ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples


        sum_val_cost = sum_val_ler = 0.
        for batch in range(num_batches_per_epoch):
            val_feed = {asr_model.inputs: val_inputs[batch * batch_size : (batch + 1) * batch_size],
                        asr_model.targets: val_targets[batch],
                        asr_model.seq_len: val_seq_len[batch * batch_size : (batch + 1) * batch_size]}
            #run_sense = session.run([dense],feed)
            val_cost, val_ler = session.run([asr_model.cost, asr_model.ler], feed_dict=val_feed)
            sum_val_cost += val_cost
            sum_val_ler += val_ler
        if curr_epoch % 10 == 0:
            saver.save(session,save_path+"/",global_step = curr_epoch)
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         sum_val_cost/num_batches_per_epoch, sum_val_ler/num_batches_per_epoch, time.time() - start))

    for batch in range(num_batches_per_epoch):
        test_feed = {asr_model.inputs: val_inputs[batch * batch_size: (batch + 1) * batch_size],
                     asr_model.targets: val_targets[batch],
                     asr_model.seq_len: val_seq_len[batch * batch_size: (batch + 1) * batch_size]}
        # Decoding
        dense = session.run(asr_model.dense, feed_dict=test_feed)
        for i in range(batch_size):
            str = decode_str(index_char, dense[i])
            print(str)



