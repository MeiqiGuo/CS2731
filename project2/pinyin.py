import utils
import collections
import pickle
import random
random.seed(1234)
import matplotlib.pyplot as plt


class Pinyin(object):
    def __init__(self, N):
        self.N = N
        self.s = '<s>'
        self.end = '</s>'
        self.space = '<space>'
        self.dict_pinyin2chinese = collections.defaultdict(list)
        self.vocab = {self.s: 0, self.end: 1}
        self.vocab_size = 2
        self.counts_ngram = {i: collections.Counter() for i in range(N)}
        self.count_unique = {i: collections.defaultdict(set) for i in range(N)}

    def pinyin2chinese(self, map_filename):
        """Get dictionary mapping pinyin to chinese characters"""
        for line in open(map_filename):
            line = line.strip("\n")
            w, pinyin = line.split(" ")
            self.dict_pinyin2chinese[pinyin].append(w)
        print("{} types of pinyin in charmap".format(len(self.dict_pinyin2chinese)))

    def get_vocab(self, train_filename, dev_filename, test_filename):
        """Get the vocabulary of han"""
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
        pickle.dump(self.vocab, open("save/han_vocab.pkl", "wb"))
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
        if self.s in self.counts_ngram[0]:
            print("Number of <s> {}".format(self.counts_ngram[0][self.s]))
        self.total_unigram = sum([count for unigram, count in self.counts_ngram[0].items()])
        print("The total number of characters in the train dataset is:{}".format(self.total_unigram))
        pickle.dump(self.counts_ngram, open("save/counts_" + str(self.N) + "gram_all.pkl", "wb"))
        pickle.dump(self.count_unique, open("save/counts_" + str(self.N) + "gram_unique.pkl", "wb"))

    def evaluator(self, filename_pinyin, filename_han, if_random=False):
        """
        Evaluate the han character prediction on dev/test data
        If if_random is True, the evaluator predicts randomly a han character from the mapping dictionary;
        ptherwise, it predicts the most possible han character based on language model.
        :param filename_pinyin:
        :param filename_han:
        :param if_random:
        :return:
        """
        total_count = 0
        acc = 0
        pinyin_data = []
        han_data = []

        given = self.start()

        for line in open(filename_pinyin):
            line = line.rstrip('\n')
            pinyin_data.append(line.split(" "))

        for line in open(filename_han):
            line = line.rstrip('\n')
            han_data.append(list(line))

        for pinyins, hans in zip(pinyin_data, han_data):
            for pinyin, han in zip(pinyins, hans):
                pred = self.predict_next(given, pinyin, if_random)
                if pred == han:
                    acc += 1
                total_count += 1
                given = self.read(given, han)

        acc = acc * 1.0 / total_count
        print("The accuracy of predicting the han character is {}.".format(acc))
        print("Total count of characters on test data are {}".format(total_count))
        return acc

    def start(self):
        """Reset the state to the initial state."""
        return tuple([self.s] * (self.N - 1))

    def read(self, given, w):
        """Read in character w, updating the state."""
        return tuple(list(given)[1:] + [w])

    def predict_next(self, given, pinyin, if_random=False):
        # If the input pinyin is space, then predict space
        if pinyin == self.space:
            return pinyin

        possible_hans = self.dict_pinyin2chinese[pinyin]

        if len(pinyin) == 1:  # The case where the pinyin is an english character
            if pinyin not in possible_hans:
                possible_hans.append(pinyin)

        assert len(possible_hans) > 0, "Error there is no possible hans for this pinyin {}".format(pinyin)

        if if_random:
            pred = possible_hans[random.randrange(len(possible_hans))]
            return pred

        max_prob = float('-inf')
        for next in possible_hans:
            p = self.prob(given, next)
            if p > max_prob:
                pred = next
                max_prob = p
        return pred

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("trainFileName", help="file name of the training han dataset")
    parser.add_argument("devFileName", help="file name of the development han dataset")
    parser.add_argument("testFileName", help="file name of the test han dataset")
    parser.add_argument("charmap_filename", help="file name of the charmap dictionary")
    parser.add_argument("devPinyinFileName", help="file name of the development pinyin dataset")
    parser.add_argument("testPinyinFileName", help="file name of the test pinyin dataset")
    parser.add_argument("N", help="Ngram number.", type=int)
    parser.add_argument("-dev", help="If True, the evaluator runs on the dev dataset; otherwise runs on test data",
                        action='store_true')
    parser.add_argument("-baseline", help="If True, the evaluator predicts randomly a han character from the mapping dictionary",
                        action='store_true')
    args = parser.parse_args()

    if args.N:
        model = Pinyin(args.N)
        model.pinyin2chinese(args.charmap_filename)
        model.get_vocab(args.trainFileName, args.devFileName, args.testFileName)
        print("{} gram language model on han with Witten-Bell smoothing are training".format(args.N))
        model.train(args.trainFileName)
        print("Finished training.")
        if args.dev:
            print("Testing on the dev dataset...")
            model.evaluator(args.devPinyinFileName, args.devFileName, args.baseline)
        else:
            print("Testing on the test dataset...")
            model.evaluator(args.testPinyinFileName, args.testFileName, args.baseline)
    else:
        accs = []
        for N in range(2, 11):
            model = Pinyin(N)
            model.pinyin2chinese(args.charmap_filename)
            model.get_vocab(args.trainFileName, args.devFileName, args.testFileName)
            print("{} gram language model on han with Witten-Bell smoothing are training".format(N))
            model.train(args.trainFileName)
            print("Finished training.")
            print("Testing on the dev dataset...")
            acc = model.evaluator(args.devPinyinFileName, args.devFileName, args.baseline)
            accs.append(acc)

        plt.figure()
        plt.plot(range(2, 11), accs)
        plt.xlabel('N')
        plt.ylabel('Accuracy')
        plt.title('Performance of different Ngram models')
        plt.savefig("save/han_ngram_acc.jpg")

