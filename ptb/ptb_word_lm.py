#encoding:utf-8
#@Time : 2018/6/22 11:11
#@Author : JackNiu

import tensorflow  as tf
import ptb_reader

flags = tf.flags
logging = tf.logging

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")

FLAGS =flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput():
    def __init__(self,config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (len(data)// batch_size -1) // num_steps
        self.input_data,self.target = ptb_reader.ptb_producer(data,batch_size,num_steps)

class PTBModel():
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",[vocab_size,hidden_size],dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        # 需要细致的讨论一下
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        with tf.variable_scope("softmax_output"):
            softmax_w = tf.get_variable(
                "softmax_w", [hidden_size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])


    def _build_rnn_graph(self, inputs, config, is_training):
        # if config.rnn_mode == CUDNN:
        #     return self._build_rnn_graph_cudnn(inputs, config, is_training)
        # else:
        return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode ==BASIC:
            return  tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,forget_bias=0.0,reuse=not is_training)
        # if config.rnn_mode == BLOCK:
        #     return tf.contrib.rnn.LSTMBlockCell(
        #         config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state

        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs=[]
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output,state

