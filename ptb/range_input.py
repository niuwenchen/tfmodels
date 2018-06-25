import tensorflow as tf
import codecs

INPUT_SIZE = 9
NUM_EXPOCHES = 5


def input_producer():
    array = codecs.open("test.txt").readlines()
    array = list(map(lambda line: line.strip(), array))
    array = tf.reshape(array,[4,9])
    i = tf.train.range_input_producer(4, num_epochs=2,shuffle=False).dequeue()
    inputs = tf.strided_slice(array, [0,i * INPUT_SIZE], [4,(i+1)*INPUT_SIZE])

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            index = 0
            while not coord.should_stop() and index < 100:
                datalines = sess.run(inputs)
                index += 1
                print("step: %d, batch data: %s" % (index, str(datalines)))
        except tf.errors.OutOfRangeError:
            print("Done traing:-------Epoch limit reached")
        except KeyboardInterrupt:
            print("keyboard interrput detected, stop training")
        finally:
            coord.request_stop()


input_producer()