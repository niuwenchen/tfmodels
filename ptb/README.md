## PTB model
building a PTB LSTM model

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
[http://arxiv.org/abs/1409.2329](http://arxiv.org/abs/1409.2329)

The hyperparameters used in the model:

* init_scale - the initial scale of the weights
* learning_rate - the initial value of the learning rate
* max\_grad\_norm - the maximum permissible norm of the gradient
* num_layers - the number of LSTM layers
* num_steps - the number of unrolled steps of LSTM
* hidden_size - the number of LSTM units
* max\_epoch - the number of epochs trained with the initial learning rate
* max\_max\_epoch - the total number of epochs for training
* keep\_prob - the probability of keeping weights in the dropout layer
* lr_decay - the decay of the learning rate for each epoch after "max\_epoch"
* batch_size - the batch size
* rnn_mode - the low level implementation of lstm cell: one of CUDNN, BASIC,or BLOCK, representing cudnn\_lsdm,basic\_lsdm,and lstm\_block\_cell classes.

The data required for this example is int the data/ dir of the PTB dataset from Tomas Micolov's webpage:

	$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
	$ tar xvf simple-examples.tgz

To run:
	
	$ python ...



### ptb_reader 文件
读取数据，将数据转换为需要的格式

原始数据转换为index:[1132, 93, ...326, 2506, 5, 0, 658]

需要的数据格式为: [batch_size,input_size] [128,28]

用队列的方式读取数据

	raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_data)  # 73760
	
    batch_len = data_len // batch_size  # 576
    data = tf.reshape(raw_data[0: batch_size * batch_len],
       [batch_size, batch_len])   # 128 x 576

    epoch_size = (batch_len - 1) // num_steps # 20
	# epoch_size 代表能从数据集中取出20个数据大块，并且这20个数据块中都是有数据的。如果epoch_size大于 20,那么20以后的数据块是空的，不能用来做运算。
	
	# 构造队列
	i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
	# 读取数据
	x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
	# 实际上是从[0,0]开始到[128,576]列的移动，每次移动num_steps个长度,此时X是[128,28]
	y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
	# y在x的基础上右移一个单位，相当于n-gram中n=1,
	# [0,1]到[128,576]列的移动

	# x和y之间的关联
	# x作为输入，y作为输出，x是前一个单词，y是后一个单词。
	
	# 注意x 和 y是iterator


	
	