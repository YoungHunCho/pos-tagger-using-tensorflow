import utils
import tensorflow as tf
import numpy as np
import os
import pickle

VOCAB_PATH = "cache/vocab.pkl"

def run():
    sess = tf.Session()

    while(True):
        sentence = input("Plese input text(q is exit): ")
        if sentence == "q": break

        textloader = utils.DataSetting("cache/vocab.pkl", sentence, n_past_words=3)

        checkpoint_file = tf.train.latest_checkpoint('checkpoints/')
        saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
        saver.restore(sess, checkpoint_file)

        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

        predicted_pos_ids = \
            sess.run(predictions, feed_dict={input_x: textloader.features})

        words = []
        for sentence_word_ids in textloader.features:
            word_id = sentence_word_ids[0]
            words.append(textloader.id_to_word[word_id])
        predicted_pos = []
        for pred_id in predicted_pos_ids:
            predicted_pos.append(textloader.id_to_pos[pred_id])

        word_pos_tuples = zip(words, predicted_pos)
        annotated_words = []
        for tup in word_pos_tuples:
            annotated_word = '%s/%s' % (tup[0], tup[1])
            annotated_words.append(annotated_word)
        annotated_sentence = ' '.join(annotated_words)

        print("The result: ")
        print(annotated_sentence)



if __name__ == "__main__":
    if not os.path.exists(VOCAB_PATH):
        print("Don't exist the vocab(pkl) file")
    else:
        run()

