from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import random
random.seed(1234)
import pickle
import numpy as np


START = '<s>'
END = '</s>'


def bootstrap_test(result1, result2, y_test, n_samples):
    N = len(y_test)
    acc1 = accuracy_score(y_test, result1)
    acc2 = accuracy_score(y_test, result2)
    delta = acc1 - acc2
    s = 0
    for iter in range(n_samples):
        y1, y2, y_true = resample(result1, result2, y_test, replace=True, n_samples=N)
        delta_sample = accuracy_score(y_true, y1) - accuracy_score(y_true, y2)
        if delta_sample > 2 * delta:
            s += 1
    p_value = 1.0 * s / n_samples
    return delta, p_value


def load_train_data(train_filename):
    """
    Append each line of characters into a long text, adding start and end symbols for each sentence.
    :param train_filename:
    :return:
    """
    train_text = []
    for line in open(train_filename):
        train_text.append(START)
        for w in line.rstrip('\n'):
            train_text.append(w)
        train_text.append(END)
    print ('Length of text: {} characters'.format(len(train_text)))
    return train_text


def load_test_data(test_filename):
    """
    A list of list of characters, adding start and end symbols for each sentence.
    :param test_filename:
    :return:
    """
    test_text = []
    for line in open(test_filename):
        sent = [START]
        for w in line.rstrip('\n'):
            sent.append(w)
        sent.append(END)
        test_text.append(sent)
    print("Number of sentences in test data: {}".format(len(test_text)))
    return test_text


def load_vocab(vocab_filename):
    """
    Dictionary of vocabulary: {v:id}
    :param vocab_filename:
    :return:
    """
    vocab = pickle.load(open(vocab_filename, "rb"))
    print ('{} unique characters in all dataset'.format(len(vocab)))
    return vocab


def text2id(text, vocab2id):
    text_as_int = np.array([vocab2id[c] for c in text])
    print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
    return text_as_int


def test2id(test, vocab2id):
    text_as_int = [[vocab2id[c] for c in sent] for sent in test]
    return text_as_int


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text



