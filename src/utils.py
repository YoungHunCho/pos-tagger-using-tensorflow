import numpy as np
from collections import Counter
import pickle
import os
import re

UNKNOWN_WORD_ID = 0
UNKNOWN_WORD = "UNK"

UNTAGGED_POS_ID = 0
UNTAGGED_POS = "UNK"

VOCAB_SIZE = 100000

class DataSetting:
    def __init__(self, vocab_path, sentences, n_past_words, tensor_path=None):
        self.n_past_words = n_past_words

        if os.path.exists(vocab_path):
            print("Load vocab")
            self.load_vocab(vocab_path)
        else:
            print("make vocab")
            self.make_vocab(sentences)
            self.save_vocab(vocab_path)

        if tensor_path is not None and os.path.exists(tensor_path):
            print("load tensor")
            self.load_tensors(tensor_path)
        else:
            print("make tensor")
            self.make_features_and_labels(sentences)
            if tensor_path is not None:
                self.save_tensors(tensor_path)

    def make_vocab(self, tagged_sentences):
        global VOCAB_SIZE
        words, pos_tags = self.split_sentence(tagged_sentences)

        word_counts = Counter(words)
        unique_pos_tags = set(pos_tags)

        words_to_keep = [t[0] for t in word_counts.most_common(VOCAB_SIZE - 1)]

        self.word_to_id = {word: i for i, word in enumerate(words_to_keep, start=1)}
        self.word_to_id[UNKNOWN_WORD] = UNKNOWN_WORD_ID

        self.pos_to_id = {pos: i for i, pos in enumerate(list(unique_pos_tags), start=1)}
        self.pos_to_id[UNTAGGED_POS] = UNTAGGED_POS_ID

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_pos = {v: k for k, v in self.pos_to_id.items()}

        self.words = words

    def load_tensors(self, tensors_path):
        with open(tensors_path, 'rb') as f:
            tensors = pickle.load(f)
        self.features = tensors[0]
        self.labels = tensors[1]

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            dicts = pickle.load(f)
        self.word_to_id = dicts[0]
        self.pos_to_id = dicts[1]
        self.id_to_word = dicts[2]
        self.id_to_pos = dicts[3]

    def save_vocab(self, vocab_filename):
        dicts = [self.word_to_id,
                 self.pos_to_id,
                 self.id_to_word,
                 self.id_to_pos]
        with open(vocab_filename, 'wb') as f:
            pickle.dump(dicts, f)

    def save_tensors(self, tensors_path):
        tensors = [self.features, self.labels]
        with open(tensors_path, 'wb') as f:
            pickle.dump(tensors, f)

    def make_features_and_labels(self, tagged_sentences):
        x , y = [], []

        for sentence in tagged_sentences.split('\n'):
            words, pos_tags = self.split_sentence(sentence)

            for j in range(len(words)):
                if len(pos_tags) != 0:
                    tag = pos_tags[j]
                    y.append(self.pos_to_id[tag])

                past_word_ids = []
                for k in range(0, self.n_past_words+1):
                    if j-k < 0: # out of bounds
                        past_word_ids.append(UNKNOWN_WORD_ID)
                    elif words[j-k] in self.word_to_id:
                        past_word_ids.append(self.word_to_id[words[j-k]])
                    else: # word not in vocabulary
                        past_word_ids.append(UNKNOWN_WORD_ID)
                x.append(past_word_ids)

        self.features = x
        self.labels = y


    def split_sentence(self, tagged_sentence):
        tagged_words = tagged_sentence.split()
        word_tag_tuples = [x.split("/") for x in tagged_words]

        words, pos_tags = [], []
        for word_tag_tuple in word_tag_tuples:
            word = word_tag_tuple[0]
            words.append(word)

            if len(word_tag_tuple) == 1:
                pos_tags.append(UNTAGGED_POS)
            else:
                tag = word_tag_tuple[1]
                pos_tags.append(tag)

        return words, pos_tags

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]
