#encoding:utf-8
#@Time : 2018/6/22 11:39
#@Author : JackNiu
import collections
import tensorflow as tf

x=['a','b','c','d','a','b','c','a','b','a','d','d','d','d']
counter = collections.Counter(x)
print(counter)
print(counter.items())
counter = sorted(counter.items(),key=lambda x:-x[1])
print(counter)
words,_=list(zip(*counter))
print(words)
print(zip(words,range(len(words))))

# print(counter)

import numpy as np


# def my_func(arg):
#     arg = tf.convert_to_tensor(arg)
#     return tf.matmul(arg, arg) + arg
#
#     # The following calls are equivalent.
#
#
# value_1 = my_func(tf.constant([[1, 2], [3, 4],[5,6]]))  # tensor
# value_2 = my_func([[1, 2], [3, 4]])  # python list
# value_3 = my_func(np.array([[1.0, 2], [3, 4]], dtype=np.float32))  # numpy arrays
#
# with tf.Session() as sess:
#     print(value_1,value_2,value_3)
#     print(sess.run(tf.size(value_1)))
#     print(value_1.shape)
#     result1, result2, result3 = sess.run([value_1, value_2, value_3])
#     print('result1 = \n%s' % (result1))
#     print('result2 = \n%s' % (result2))
#     print('result3 = \n%s' % (result3))