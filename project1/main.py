# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords
import csv
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.utils import resample
import argparse
import matplotlib.pyplot as plt
import pickle


def load(filename):
    comments = []
    scores = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        # don't read header
        next(csvReader)
        for row in csvReader:
            comments.append(row[5].lower())
            scores.append(int(row[8][0]))
    return comments, scores


def prepross(comments, min_freq=2, new_version=False):
    # Count Vocab
    counter = Counter()
    # Tokenize
    tokenized_comments = []
    for comment in comments:
        tokens = nltk.word_tokenize(comment)
        tokenized_comments.append(tokens)
        counter.update(tokens)
    print("Total vocabulary size is {}.".format(len(counter)))
    #print("Most frequent token {}.".format(counter.most_common(50)))
    vocab = {}
    idx = 0
    if new_version:
        for word, count in counter.items():
            if count >= min_freq and word not in stopWords and len(word) > 2:
                vocab[word] = idx
                idx += 1
    else:
        for word, count in counter.items():
            if count >= min_freq and word not in stopWords:
                vocab[word] = idx
                idx += 1
    return tokenized_comments, vocab


def getFeatures(comments, vocab, new_version=False):
    # the index of len(vocab) is the id of UNK
    id_unk = len(vocab)
    features = []
    for comment in comments:
        X = [0] * (id_unk + 1)
        if new_version:
            for token in comment:
                if token in vocab:
                    X[vocab[token]] += 1
                elif token not in stopWords and len(token) > 2:
                    X[id_unk] += 1
        else:
            for token in comment:
                if token in vocab:
                    X[vocab[token]] += 1
                elif token not in stopWords:
                    X[id_unk] += 1
        features.append(X)
    return features


def train(X, y, n_folds=5, print_result=True):
    N = len(y)
    X = np.array(X)
    y = np.array(y)
    kf = KFold(N, n_folds=n_folds, shuffle=True, random_state=42)
    accs = []
    y_pred_total = []
    y_test_total = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_pred_total += list(y_pred)
        y_test_total += list(y_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        if print_result:
            report = classification_report(y_test, y_pred)
            print("Metric tables for each toxic level based on the cross-validation test dataset:")
            print(report)
    if print_result:
        print("Accuracies for {0} fold cross-validation: {1}.".format(n_folds, accs))
        print("Average accuracy: {}.".format(sum(accs) / n_folds))
    return y_pred_total, y_test_total, sum(accs) / n_folds


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


def featureSelection1(features, dim=200):
    svd = TruncatedSVD(dim, random_state=42)
    features = svd.fit_transform(features)
    return features


def featureSelection2(features, dim=200):
    pca = PCA(n_components=dim, random_state=42)
    features = pca.fit_transform(features)
    return features


def featureSelection3(features, scores, dim=200, k=30):
    chi2_test, _ = chi2(features, scores)
    sorted_idx = sorted(range(len(chi2_test)), key=lambda x: chi2_test[x], reverse=True)
    selected_tokens = [vocab_inv[i] for i in sorted_idx[:k]]
    model = SelectKBest(chi2, k=dim)
    features = model.fit_transform(features, scores)
    #selected_tokens2 = [vocab_inv[i] for i in model.get_support(indices=True)]
    #print(selected_tokens2)
    return features, selected_tokens


def featureSelection4(features, scores, dim=200, k=30):
    multual_test = mutual_info_classif(features, scores, random_state=42)
    sorted_idx = sorted(range(len(multual_test)), key=lambda x: multual_test[x], reverse=True)
    selected_tokens = [vocab_inv[i] for i in sorted_idx[:k]]
    features = SelectKBest(mutual_info_classif, k=dim).fit_transform(features, scores)
    return features, selected_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fileName", help="configuration file name of the dataset")
    parser.add_argument("step", help="variable for step", choices=[2, 4, 5], type=int)
    args = parser.parse_args()

    stopWords = set(stopwords.words('english'))
    stopWords.update([".", ",", "'s", "'", ")", "(", "-", ":"])
    comments, scores = load(filename=args.fileName)
    #counter = Counter(scores)
    #print(counter)
    #print(len(scores))

    # Step 2
    if args.step == 2:
        tokenized_comments, vocab = prepross(comments, min_freq=2, new_version=False)
        vocab_inv = {id: w for w, id in vocab.items()}
        vocab_inv[len(vocab)] = "UNK"
        print("Filtered vocabulary size is {}".format(len(vocab)))
        features = getFeatures(tokenized_comments, vocab, new_version=False)
        train(features, scores)

    # Step 4
    if args.step == 4:
        tokenized_comments, vocab1 = prepross(comments, min_freq=2, new_version=False)
        vocab_inv1 = {id: w for w, id in vocab1.items()}
        vocab_inv1[len(vocab1)] = "UNK"
        print("Filtered vocabulary size for old model is {}".format(len(vocab1)))
        features1 = getFeatures(tokenized_comments, vocab1, new_version=False)

        tokenized_comments, vocab2 = prepross(comments, min_freq=2, new_version=True)
        vocab_inv2 = {id: w for w, id in vocab2.items()}
        vocab_inv2[len(vocab2)] = "UNK"
        print("Filtered vocabulary size for new model is {}".format(len(vocab2)))
        features2 = getFeatures(tokenized_comments, vocab2, new_version=True)

        result1, y_test_total, acc1 = train(features1, scores, print_result=False)
        print("Accuracy of old model is {}.".format(acc1))
        result2, _, acc2 = train(features2, scores, print_result=False)
        print("Accuracy of new model is {}.".format(acc2))
        if acc2 > acc1:
            delta, p_value = bootstrap_test(result2, result1, y_test_total, n_samples=100)
            print("New model is better than old model by {0} in accuracy with p_value {1}".format(delta, p_value))
        else:
            delta, p_value = bootstrap_test(result1, result2, y_test_total, n_samples=100)
            print("Old model is better than new model by {0} in accuracy with p_value {1}".format(delta, p_value))

    # Step 5
    if args.step == 5:
        tokenized_comments, vocab = prepross(comments, min_freq=2, new_version=True)
        vocab_inv = {id: w for w, id in vocab.items()}
        vocab_inv[len(vocab)] = "UNK"
        print("Filtered vocabulary size is {}".format(len(vocab)))
        features = getFeatures(tokenized_comments, vocab, new_version=True)
        svd = []
        pca = []
        chi = []
        mi = []
        dims = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for dim in dims:
            features1 = featureSelection1(features, dim=dim)
            _, _, acc1 = train(features1, scores, print_result=False)
            svd.append(acc1)
            features2 = featureSelection2(features, dim=dim)
            _, _, acc2 = train(features2, scores, print_result=False)
            pca.append(acc2)
            features3, top_features1 = featureSelection3(features, scores, dim=dim)
            _, _, acc3 = train(features3, scores, print_result=False)
            chi.append(acc3)
            features4, top_features2 = featureSelection4(features, scores, dim=dim)
            _, _, acc4 = train(features4, scores, print_result=False)
            mi.append(acc4)

        print("Top discriminant tokens by Chi2 test are {}.".format(top_features1))
        print("Top discriminant tokens by Mutual Information test are {}.".format(top_features2))
        print("Accuracies of SVD dimension reduction method is {}.".format(svd))
        print("Accuracies of PCA dimension reduction method is {}.".format(pca))
        print("Accuracies of Chi2 dimension reduction method is {}.".format(chi))
        print("Accuracies of Mutual Information dimension reduction method is {}.".format(mi))
        pickle.dump([svd, pca, chi, mi], open("acc_step5.pkl", 'wb'))

        #svd, pca, chi, mi = pickle.load(open("acc_step5.pkl", 'rb'))
        #dims = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        # Plot
        plt.figure()
        plt.plot(dims, svd, label='SVD', color='deeppink', marker='*')
        plt.plot(dims, pca, label='PCA', color='navy', marker='*')
        plt.plot(dims, chi, label='Chi2', color='aqua', marker='*')
        plt.plot(dims, mi, label='MI', color='darkorange', marker='*')
        plt.plot(dims, [0.768] * len(dims), label='Baseline', color='black', linestyle='--')
        plt.xlim([50, 1000])
        plt.ylim([0.65, 0.85])
        plt.xlabel('Feature Dimension')
        plt.ylabel('Accuracy')
        plt.title('Performance of different methods for feature dimension reduction')
        plt.legend(loc="lower right")
        plt.savefig("fig_step5.jpg")





