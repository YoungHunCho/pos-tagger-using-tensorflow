import tensorflow as tf

class Tagger(object):
    def __init__(self, vocab_size, embedding_size, n_past_words, n_pos_tags):
        self.input_x = tf.placeholder(
            tf.int32, [None, n_past_words + 1], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))

        with tf.name_scope("model"):
            self.word_matrix = \
                tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

            new_shape = [-1, (n_past_words + 1) * embedding_size]
            self.feature_vector = tf.reshape(self.word_matrix, new_shape)


            feature_vector_size = int(self.feature_vector.shape[1])
            h_size = 100
            w1 = tf.Variable(tf.truncated_normal([feature_vector_size, h_size], stddev=0.1))
            self.h = tf.nn.relu(tf.matmul(self.feature_vector, w1))

            _w1 = tf.Variable(tf.truncated_normal([h_size, h_size]))
            _h1 = tf.nn.relu(tf.matmul(self.h, _w1))

            _w2 = tf.Variable(tf.truncated_normal([h_size, h_size]))
            _h2 = tf.nn.relu(tf.matmul(_h1, _w2))

            _w3 = tf.Variable(tf.truncated_normal([h_size, h_size]))
            _h3 = tf.nn.relu(tf.matmul(_h2, _w3))

            _w4 = tf.Variable(tf.truncated_normal([h_size, h_size]))
            _h4 = tf.nn.relu(tf.matmul(_h3, _w4))

            self.w2 = tf.Variable(tf.truncated_normal([h_size, n_pos_tags], stddev=0.1))
            self.logits = tf.matmul(_h4, self.w2)
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.logits))

        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(
                self.logits, axis=1, name='predictions')
            correct_prediction = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))