import re
from collections import Counter
import numpy as np


def clean_str(string):
    """
        String cleaning
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        :param string: sentences
        :return: cleared sentence
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def preprocess_data(sentence_train, label_train, sentence_test, label_test):
    """
        Converting Word to Number while creating our own dictionary
        :param sentence_train: train dataset inputs
        :param label_train: train labels
        :param sentence_test: test dataset inputs
        :param label_test: test labels
        :return:
    """
    word_list = []

    # Create vocabulary based on the words in dataset
    for sent in sentence_train:
        for word in sent.lower().split():
            word = clean_str(word)
            if word != '' and len(word) > 2:
                word_list.append(word)

    # Counting Words
    count = Counter(word_list)

    # Sorting Words based on the number of usage
    corpus_ = sorted(count, key=count.get, reverse=True)

    # Creating Dictionary
    word_dictionary = {w: i + 1 for i, w in enumerate(corpus_)}

    # Converting Word to Number and Get Labels for train set
    final_sentence_train, final_label_train, final_sentence_test, final_label_test = [], [], [], []
    for i, (sent, lbl) in enumerate(zip(sentence_train, label_train)):
        cleaned_sent = []
        for word in sent.lower().split():
            if clean_str(word) in word_dictionary.keys():
                cleaned_sent.append(word_dictionary[clean_str(word)])
        final_sentence_train.append(cleaned_sent)
        final_label_train.append(lbl)

    # Converting Word to Number and Get Labels for test set
    for i, (sent, lbl) in enumerate(zip(sentence_test, label_test)):
        cleaned_sent = []
        for word in sent.lower().split():
            if clean_str(word) in word_dictionary.keys():
                cleaned_sent.append(word_dictionary[clean_str(word)])
        final_sentence_test.append(cleaned_sent)
        final_label_test.append(lbl)

    # Get Length of Dictionary
    length_of_dictionary = len(word_dictionary) + 1

    return np.array(final_sentence_train), np.array(final_label_train), np.array(final_sentence_test), np.array(
        final_label_test), length_of_dictionary


def padding(sentences, seq_len):
    """
        Pad the sentences to sequence length
        :param sentences: input sentences
        :param seq_len: sequence length
        :return: padded sentences
    """
    padded_sentences = np.zeros((len(sentences), seq_len), dtype=int)
    for i, sentence in enumerate(sentences):
        if len(sentence) != 0:
            padded_sentences[i, -len(sentence):] = np.array(sentence)[:seq_len]
    return padded_sentences
