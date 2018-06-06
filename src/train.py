import os
import time
import tensorflow as tf
import random
import utils
import model

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'
CACHE_DIR = 'cache'
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100


def load_data(data_path, n_past_words, test_proportion, batch_size,
              n_epochs):
    tagged_sentences = ""
    with open(data_path, 'r', encoding='utf8') as f:
        tagged_sentences = f.read()
    tagged_sentences = tagged_sentences.split("\n")
    random.shuffle(tagged_sentences)
    tagged_sentences = "\n".join(tagged_sentences)

    textloader = utils.DataSetting("cache/vocab.pkl", tagged_sentences, n_past_words, "cache/tensors.pkl")

    x = textloader.features
    y = textloader.labels
    n_pos_tags = len(textloader.pos_to_id)

    idx = int(test_proportion * len(x))
    x_test, x_train = x[:idx], x[idx:]
    y_test, y_train = y[:idx], y[idx:]

    train_batches = utils.batch_iter(
        list(zip(x_train, y_train)), batch_size, n_epochs)
    test_data = {'x': x_test, 'y': y_test}

    return (train_batches, test_data, n_pos_tags)


def model_init(vocab_size, embedding_size, n_past_words, n_pos_tags):
    pos_tagger = model.Tagger(vocab_size, embedding_size, n_past_words, n_pos_tags)

    global_step = tf.Variable(
        initial_value=0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(pos_tagger.loss, global_step=global_step)

    return pos_tagger, train_op, global_step


def logging_init(model, graph):
    train_loss = tf.summary.scalar("train_loss", model.loss)
    train_accuracy = tf.summary.scalar("train_accuracy", model.accuracy)

    train_summary_ops = tf.summary.merge([train_loss, train_accuracy])

    test_loss = tf.summary.scalar("test_loss", model.loss)
    test_accuracy = tf.summary.scalar("test_accuracy", model.accuracy)
    test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

    timestamp = int(time.time())
    run_log_dir = os.path.join(LOGS_DIR, str(timestamp))
    os.makedirs(run_log_dir)

    summary_writer = tf.summary.FileWriter(run_log_dir, graph)

    return train_summary_ops, test_summary_ops, summary_writer


def checkpointing_init():
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    return saver


def step(sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer,
         train):
    feed_dict = {model.input_x: x, model.input_y: y}

    if train:
        step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops,
                                                      feed_dict)
    else:
        step, loss, accuracy, summaries = sess.run(standard_ops + test_ops,
                                                   feed_dict)

    print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
    summary_writer.add_summary(summaries, step)


def main():
    test_proportion = 0.1
    n_past_words = 3
    batch_size = 256
    n_epochs = 50
    embedding_dim = 50


    sess = tf.Session()

    train_batches, test_data, n_pos_tags = load_data("data/tagged.txt",
         n_past_words,
        test_proportion, batch_size, n_epochs)
    x_test = test_data['x']
    print(x_test[:10])
    y_test = test_data['y']
    pos_tagger, train_op, global_step = model_init(
        utils.VOCAB_SIZE, embedding_dim, n_past_words, n_pos_tags)
    train_summary_ops, test_summary_ops, summary_writer = logging_init(
        pos_tagger, sess.graph)
    saver = checkpointing_init()

    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    standard_ops = [global_step, pos_tagger.loss, pos_tagger.accuracy]
    train_ops = [train_op, train_summary_ops]
    test_ops = [test_summary_ops]

    for batch in train_batches:
        x_batch, y_batch = zip(*batch)
        step(
            sess,
            pos_tagger,
            standard_ops,
            train_ops,
            test_ops,
            x_batch,
            y_batch,
            summary_writer,
            train=True)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % EVALUATE_EVERY == 0:
            print("\nEvaluation:")
            step(
                sess,
                pos_tagger,
                standard_ops,
                train_ops,
                test_ops,
                x_test,
                y_test,
                summary_writer,
                train=False)
            print("")

        if current_step % CHECKPOINT_EVERY == 0:
            prefix = os.path.join(CHECKPOINT_DIR, 'model')
            path = saver.save(sess, prefix, global_step=current_step)
            print("Saved model checkpoint to '%s'" % path)

if __name__ == "__main__":
    main()