#encoding:utf-8
#@Time : 2018/6/22 11:24
#@Author : JackNiu

"""Utilities for parsing PTB text files."""
import collections
import os
import sys

import tensorflow as tf
Py3 = sys.version_info[0] == 3


def _read_words(filename):
    with tf.gfile.GFile(filename,'r') as f:
        if Py3:
            return f.read().replace('\n',"<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n","<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(),key=lambda x:-x[1])
    words,_ = list(zip(*count_pairs))
    word_to_id = dict(zip(words,range(len(words))))
    return word_to_id

# _build_vocab('./simple-examples/data/ptb.valid.txt')

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word]  for word in data  if word in word_to_id]

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_size =len(word_to_id)
    return train_data,valid_data,test_data,vocabulary_size


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """
    Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    12,10
    """
    with tf.name_scope(name,"PTBEncoder",[raw_data,batch_size,num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)

        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        # sess = tf.Session()
        # print(sess.run(epoch_size)) # epoch_size = 20 代表 取20个数据块？但是最后需要的是batch_size=128 就是取128个数据块
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

_,valid_data,_,_ = ptb_raw_data('./simple-examples/data/')
print(valid_data)
print(len(valid_data))
with tf.Session() as sess:
    x,y =ptb_producer(valid_data,128,28)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        index = 0
        while not coord.should_stop() and index < 20:
            x1 = sess.run(x)
            # y1= sess.run(y)
            index += 1
            print("step: %d, batch data: %s" % (index, str(x1)))
            # print("step: %d, batch data: %s"%(index,str(y1)))
    except tf.errors.OutOfRangeError:
        print("Done traing:-------Epoch limit reached")
    except KeyboardInterrupt:
        print("keyboard interrput detected, stop training")
    finally:
        coord.request_stop()




