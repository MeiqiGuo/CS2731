import collections
import pickle
import random
random.seed(1234)
import matplotlib.pyplot as plt
import numpy as np
import utils


class Ngram(object):
    """An Ngram language model class."""

    def __init__(self, N):
        self.counts_ngram = collections.Counter()
        self.counts_pre_ngram = collections.Counter()
        self.N = N
        self.s = '<s>'
        self.end = '</s>'

    def get_vocab(self, train_filename, dev_filename, test_filename):
        """Get the vocabulary"""
        self.vocab = {self.s: 0, self.end: 1}
        self.vocab_size = 2
        for line in open(train_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(dev_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(test_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        print("The vocabulary size is {}.".format(self.vocab_size))
        pickle.dump(self.vocab, open("save/vocab.pkl", "wb"))
        self.vocab_reverse = {i: w for w, i in self.vocab.items()}

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            line = list(line.rstrip('\n'))
            line = [self.s] * (self.N - 1) + line + [self.end]
            ngrams = [tuple(line[i:i + self.N]) for i in range(len(line) - self.N + 1)]
            pre_ngrams = [tuple(line[i:i + self.N - 1]) for i in range(len(line) - self.N + 1)]
            self.counts_ngram.update(ngrams)
            self.counts_pre_ngram.update(pre_ngrams)
        pickle.dump(self.counts_pre_ngram, open("save/counts_pre_" + str(self.N) + "gram.pkl", "wb"))
        pickle.dump(self.counts_ngram, open("save/counts_" + str(self.N) + "gram.pkl", "wb"))

    def evaluator(self, filename):
        """Evaluate the Ngram model on dev/test data"""
        total_count = 0
        acc = 0
        y_true = []
        y_pred = []
        for line in open(filename):
            given = self.start()
            line = list(line.rstrip('\n')) + [self.end]
            for w in line:
                pred = self.predict_next(given)
                y_pred.append(pred)
                y_true.append(w)
                if pred == w:
                    acc += 1
                total_count += 1
                given = self.read(given, w)
        acc = acc * 1.0 / total_count
        print("The accuracy of predicting the next character is {}.".format(acc))
        return acc, y_pred, y_true

    def start(self):
        """Reset the state to the initial state."""
        return tuple([self.s] * (self.N - 1))

    def read(self, given, w):
        """Read in character w, updating the state."""
        return tuple(list(given)[1:] + [w])

    def prob(self, given, next):
        """Return the probability of the next character being w given the
        current state."""
        ngram = given + (next, )
        if ngram in self.counts_ngram and given in self.counts_pre_ngram:
            return self.counts_ngram[ngram] / self.counts_pre_ngram[given]
        elif given in self.counts_pre_ngram:
            return 0
        else:
            return 1.0 / self.vocab_size

    def predict_next(self, given):
        if given not in self.counts_pre_ngram:
            # If it doesn't exist in counter, then I randomly choose one character from vocabulary
            pred = self.vocab_reverse[random.randrange(self.vocab_size)]
        else:
            max_prob = float('-inf')
            for next in self.vocab:
                p = self.prob(given, next)
                if p > max_prob:
                    pred = next
                    max_prob = p
        return pred


class InterpolationNgram(object):
    """An Ngram language model class with linear interpolation smoothing (Jelinek-Mercer smoothing).
       Discount parameter is assumed context-independent here.
    """

    def __init__(self, N):
        """
        :param N: The N number of ngram model
        """
        self.counts_ngram = {i: collections.Counter() for i in range(N)}
        self.N = N
        self.s = '<s>'
        self.end = '</s>'

    def get_vocab(self, train_filename, dev_filename, test_filename):
        """Get the vocabulary"""
        self.vocab = {self.s: 0, self.end: 1}
        self.vocab_size = 2
        for line in open(train_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(dev_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(test_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        print("The vocabulary size is {}.".format(self.vocab_size))
        pickle.dump(self.vocab, open("save/vocab.pkl", "wb"))
        self.vocab_reverse = {i: w for w, i in self.vocab.items()}

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            line = list(line.rstrip('\n'))
            for n in range(1, self.N + 1):
                new_line = [self.s] * (n - 1) + line + [self.end]
                ngrams = [tuple(new_line[i:i + n]) for i in range(len(new_line) - n + 1)]
                self.counts_ngram[n - 1].update(ngrams)
        self.total_unigram = sum([count for unigram, count in self.counts_ngram[0].items()])
        print("The total number of characters in the train dataset is:{}".format(self.total_unigram))
        pickle.dump(self.counts_ngram, open("save/counts_" + str(self.N) + "gram_all.pkl", "wb"))

    def evaluator(self, filename, discount):
        """Evaluate the Ngram model on dev/test data"""
        total_count = 0
        acc = 0
        y_true = []
        y_pred = []
        for line in open(filename):
            given = self.start()
            line = list(line.rstrip('\n')) + [self.end]
            for w in line:
                pred = self.predict_next(given, discount)
                y_pred.append(pred)
                y_true.append(w)
                if pred == w:
                    acc += 1
                total_count += 1
                given = self.read(given, w)
        acc = acc * 1.0 / total_count
        print("The accuracy of predicting the next character is {}.".format(acc))
        return acc, y_pred, y_true

    def start(self):
        """Reset the state to the initial state."""
        return tuple([self.s] * (self.N - 1))

    def read(self, given, w):
        """Read in character w, updating the state."""
        return tuple(list(given)[1:] + [w])

    def prob(self, given, next, discount):
        """Return the probability of the next character being w given the
        current state.
        :param given: the given n-1 previous characters
        :param next: the character whose probability is to be estimated
        :param discount: lambda for interpolation smoothing. Refer to README for more detail.
        """
        ngram = given + (next, )
        # Initialized by Pr_us
        if ngram[self.N - 1] in self.counts_ngram[0]:
            p = discount * self.counts_ngram[0][ngram[self.N - 1]] / self.total_unigram + (1 - discount) / self.vocab_size
        else:
            p = 0 + (1 - discount) / self.vocab_size
        # Compute Pr_ngram by interpolation
        for n in range(2, self.N + 1):
            if ngram[self.N - n:] in self.counts_ngram[n - 1] and ngram[self.N - n:-1] in self.counts_ngram[n - 2]:
                p = discount * self.counts_ngram[n - 1][ngram[self.N - n:]] / self.counts_ngram[n - 2][ngram[self.N - n:-1]] \
                    + (1 - discount) * p
            elif ngram[self.N - n:-1] in self.counts_ngram[n - 2]:
                p = 0 + (1 - discount) * p
            else:
                p = 1.0 / self.vocab_size + (1 - discount) * p
        return p

    def predict_next(self, given, discount):
        max_prob = float('-inf')
        for next in self.vocab:
            p = self.prob(given, next, discount)
            if p > max_prob:
                pred = next
                max_prob = p
        return pred


class WittenBellNgram(object):
    """An Ngram language model class with Witten-Bell smoothing.
       Discount parameters are defined by the context frequency No need to tune these parameters.
    """
    def __init__(self, N):
        """
        :param N: The N number of ngram model
        """
        self.counts_ngram = {i: collections.Counter() for i in range(N)}
        self.count_unique = {i: collections.defaultdict(set) for i in range(N)}
        self.N = N
        self.s = '<s>'
        self.end = '</s>'

    def get_vocab(self, train_filename, dev_filename, test_filename):
        """Get the vocabulary"""
        self.vocab = {self.s: 0, self.end: 1}
        self.vocab_size = 2
        for line in open(train_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(dev_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        for line in open(test_filename):
            for w in line.rstrip('\n'):
                if w in self.vocab:
                    pass
                else:
                    self.vocab[w] = self.vocab_size
                    self.vocab_size += 1
        print("The vocabulary size is {}.".format(self.vocab_size))
        pickle.dump(self.vocab, open("save/vocab.pkl", "wb"))
        self.vocab_reverse = {i: w for w, i in self.vocab.items()}

    def train(self, filename):
        """Train the model on a text file."""

        for line in open(filename):
            line = list(line.rstrip('\n'))
            for n in range(1, self.N + 1):
                new_line = [self.s] * (n - 1) + line + [self.end]
                ngrams = [tuple(new_line[i:i + n]) for i in range(len(new_line) - n + 1)]
                self.counts_ngram[n - 1].update(ngrams)
                for ngram in ngrams:
                    self.count_unique[n - 1][ngram[:-1]].add(ngram[-1])
        self.total_unigram = sum([count for unigram, count in self.counts_ngram[0].items()])
        print("The total number of characters in the train dataset is:{}".format(self.total_unigram))
        pickle.dump(self.counts_ngram, open("save/counts_" + str(self.N) + "gram_all.pkl", "wb"))
        pickle.dump(self.count_unique, open("save/counts_" + str(self.N) + "gram_unique.pkl", "wb"))
        """
        self.total_unigram = 31631950
        self.counts_ngram = pickle.load(open("save/counts_" + str(self.N) + "gram_all.pkl", "rb"))
        self.count_unique = pickle.load(open("save/counts_" + str(self.N) + "gram_unique.pkl", "rb"))
        """

    def evaluator(self, filename):
        """Evaluate the Ngram model on dev/test data"""
        total_count = 0
        acc = 0
        y_pred = []
        y_true = []
        for line in open(filename):
            given = self.start()
            line = list(line.rstrip('\n')) + [self.end]
            for w in line:
                pred = self.predict_next(given)
                y_pred.append(pred)
                y_true.append(w)
                if pred == w:
                    acc += 1
                total_count += 1
                given = self.read(given, w)
        acc = acc * 1.0 / total_count
        print("The accuracy of predicting the next character is {}.".format(acc))
        return acc, y_pred, y_true

    def start(self):
        """Reset the state to the initial state."""
        return tuple([self.s] * (self.N - 1))

    def read(self, given, w):
        """Read in character w, updating the state."""
        return tuple(list(given)[1:] + [w])

    def prob(self, given, next):
        """Return the probability of the next character being w given the
        current state.
        :param given: the given n-1 previous characters
        :param next: the character whose probability is to be estimated
        """
        ngram = given + (next, )
        # Initialized by Pr_us
        n_unique = len(self.count_unique[0][()])  # the number of unique words that follow the history
        discount = self.total_unigram / (self.total_unigram + n_unique)
        if ngram[self.N - 1] in self.counts_ngram[0]:
            p = discount * self.counts_ngram[0][ngram[self.N - 1]] / self.total_unigram + (1 - discount) / self.vocab_size
        else:
            p = 0 + (1 - discount) / self.vocab_size
        # Compute Pr_ngram by interpolation
        for n in range(2, self.N + 1):
            n_unique = len(self.count_unique[n - 1][ngram[self.N - n:-1]])
            if ngram[self.N - n:] in self.counts_ngram[n - 1] and ngram[self.N - n:-1] in self.counts_ngram[n - 2]:
                n_all = self.counts_ngram[n - 2][ngram[self.N - n:-1]]  # total count of words that follow the history
                discount = n_all / (n_all + n_unique)
                p = discount * self.counts_ngram[n - 1][ngram[self.N - n:]] / self.counts_ngram[n - 2][ngram[self.N - n:-1]] \
                    + (1 - discount) * p
            elif ngram[self.N - n:-1] in self.counts_ngram[n - 2]:
                n_all = self.counts_ngram[n - 2][ngram[self.N - n:-1]]  # total count of words that follow the history
                discount = n_all / (n_all + n_unique)
                p = 0 + (1 - discount) * p
            else:
                p = p
        return p

    def predict_next(self, given):
        max_prob = float('-inf')
        for next in self.vocab:
            p = self.prob(given, next)
            if p > max_prob:
                pred = next
                max_prob = p
        return pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("trainFileName", help="file name of the training dataset")
    parser.add_argument("devFileName", help="file name of the development dataset")
    parser.add_argument("testFileName", help="file name of the test dataset")
    parser.add_argument("N", help="Ngram number. If 0, then run N from 2 to 10.", type=int)
    parser.add_argument("step", help="Step number", type=int)
    parser.add_argument("-wb", help="If True, use Witten-Bell smoothing; otherwise use Jelinek-Mercer smoothing", action='store_true')
    parser.add_argument("-dev", help="If True, the evaluator runs on the dev dataset; otherwise runs on test data", action='store_true')
    parser.add_argument("-comp", help="If True, run bootstrap test for comparing models", action='store_true')
    parser.add_argument("--discount", help="Discount parameter", default=None, type=float)
    args = parser.parse_args()

    N = args.N
    trainFileName = args.trainFileName
    devFileName = args.devFileName
    testFileName = args.testFileName
    step = args.step
    discount = args.discount

    if step == 1:
        if N == 0:
            accs = []
            for N in range(2, 11):
                ngram_model = Ngram(N)
                ngram_model.get_vocab(trainFileName, devFileName, testFileName)
                print("{} gram models are training".format(N))
                ngram_model.train(trainFileName)
                print("{} gram models are trained. Testing on the dev dataset".format(N))
                acc, _, _ = ngram_model.evaluator(devFileName)
                accs.append(acc)
            pickle.dump(accs, open("save/accs_ngram.pkl", "wb"))
            plt.figure()
            plt.plot(range(2, 11), accs)
            plt.xlabel('N')
            plt.ylabel('Accuracy')
            plt.title('Performance of different Ngram models')
            plt.savefig("save/ngram_acc.jpg")
        else:
            ngram_model = Ngram(N)
            ngram_model.get_vocab(trainFileName, devFileName, testFileName)
            print("{} gram models are training".format(N))
            ngram_model.train(trainFileName)
            print("{} gram models are trained.".format(N))
            if args.dev:
                print("Testing on the dev dataset...")
                ngram_model.evaluator(devFileName)
            else:
                print("Testing on the test dataset...")
                ngram_model.evaluator(testFileName)

    if step == 2:
        if args.comp:
            print("6-gram model without smoothing is training...")
            model1 = Ngram(6)
            model1.get_vocab(trainFileName, devFileName, testFileName)
            model1.train(trainFileName)
            print("Model is testing on the test dataset...")
            acc1, y_pred1, y_true = model1.evaluator(testFileName)

            print("9-gram model with Jelinek-Mercer smoothing is training...")
            model2 = InterpolationNgram(9)
            model2.get_vocab(trainFileName, devFileName, testFileName)
            model2.train(trainFileName)
            print("Model is testing on the test dataset...")
            acc2, y_pred2, _ = model2.evaluator(testFileName, 0.95)

            print("9-gram model with Witten-Bell smoothing is training...")
            model3 = WittenBellNgram(9)
            model3.get_vocab(trainFileName, devFileName, testFileName)
            model3.train(trainFileName)
            print("Model is testing on the test dataset...")
            acc3, y_pred3, _ = model3.evaluator(testFileName)

            print("Bootstrap testing...")
            delta, p_value = utils.bootstrap_test(y_pred2, y_pred1, y_true, n_samples=10)
            print("9-gram model with Jelinek-Mercer smoothing is better than 6-gram model without smoothing by {0} in accuracy with p_value {1}".format(delta, p_value))

            delta, p_value = utils.bootstrap_test(y_pred3, y_pred1, y_true, n_samples=10)
            print("9-gram model with Witten-Bell smoothing is better than 6-gram model without smoothing by {0} in accuracy with p_value {1}".format(delta, p_value))

            delta, p_value = utils.bootstrap_test(y_pred3, y_pred2, y_true, n_samples=10)
            print("9-gram model with Witten-Bell smoothing is better than Jelinek-Mercer smoothing by {0} in accuracy with p_value {1}".format(delta, p_value))
        else:
            if args.wb:
                ngram_model = WittenBellNgram(N)
                ngram_model.get_vocab(trainFileName, devFileName, testFileName)
                print("{} gram models with Witten-Bell smoothing are training".format(N))
                ngram_model.train(trainFileName)
                print("{} gram models are trained.".format(N))
                if not discount:
                    if args.dev:
                        print("Testing on the dev dataset...")
                        ngram_model.evaluator(devFileName)
                    else:
                        print("Testing on the test dataset...")
                        ngram_model.evaluator(testFileName)
                else:
                    print("Wrong input: no predefined discount for Witten-Bell smoothing.")
            else:
                ngram_model = InterpolationNgram(N)
                ngram_model.get_vocab(trainFileName, devFileName, testFileName)
                print("{} gram models with Jelinek-Mercer smoothing are training".format(N))
                ngram_model.train(trainFileName)
                print("{} gram models are trained.".format(N))
                if discount:
                    if args.dev:
                        print("Testing on the dev dataset...")
                        ngram_model.evaluator(devFileName, discount)
                    else:
                        print("Testing on the test dataset...")
                        ngram_model.evaluator(testFileName, discount)
                else:
                    if args.dev:
                        print("Testing on the dev dataset for choosing discount parameter...")
                        accs = []
                        #x = np.linspace(0.1, 1, 10)
                        x = np.linspace(0.6, 0.9, 16)
                        for discount in x:
                            acc, _, _ = ngram_model.evaluator(devFileName, discount)
                            accs.append(acc)
                        pickle.dump(accs, open("save/accs_ngram_JMsmoothing.pkl", "wb"))
                        plt.figure()
                        plt.plot(x, accs)
                        plt.xlabel('Discount parameter')
                        plt.ylabel('Accuracy')
                        plt.title('Performance of different Jelinek-Mercer smoothing discount value for {}gram model'.format(N))
                        plt.savefig("save/ngram_acc_JMsmoothing.jpg")

                    else:
                        print("Wrong input: cannot have dev=False and discount=None.")







