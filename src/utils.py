import numpy as np
from collections import Counter
import re

UNKNOWN_WORD_ID = 0
UNKNOWN_WORD = "UNK"

UNTAGGED_POS_ID = 0
UNTAGGED_POS = "UNK"

VOCAB_SIZE = 50000

class DataSetting:
    def __init__(self, sentences, n_past_words):
        self.n_past_words = n_past_words

        print("make vocab")
        self.make_vocab(sentences)
        
        print("make features and labels")
        self.make_features_and_labels(sentences)


    def make_vocab(self, tagged_sentences):
        global VOCAB_SIZE
        words, pos_tags = self.split_sentence(tagged_sentences)

        word_counts = Counter(words)
        unique_pos_tags = set(pos_tags)

        VOCAB_SIZE = len(word_counts) // 4 * 3
        words_to_keep = \
            [t[0] for t in word_counts.most_common(VOCAB_SIZE - 1)]

        self.word_to_id = \
            {word: i for i, word in enumerate(words_to_keep, start=1)}
        # words not in the vocabulary will be mapped to this word
        self.word_to_id[UNKNOWN_WORD] = UNKNOWN_WORD_ID # = 0

        self.pos_to_id = \
            {pos: i for i, pos in enumerate(list(unique_pos_tags), start=1)}
        self.pos_to_id[UNTAGGED_POS] = UNTAGGED_POS_ID # = 0

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_pos = {v: k for k, v in self.pos_to_id.items()}

        self.words = words

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
