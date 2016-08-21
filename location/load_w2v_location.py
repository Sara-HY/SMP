# encoding: utf-8
import numpy as np
import cPickle
from collections import defaultdict
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def build_data_cv(data_folder, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    northeastFile = data_fold[0]
    northChinaFile = data_fold[1]
    eastChinaFile = data_fold[2]
    centralChinaFile = data_fold[3]
    southChinaFile = data_fold[4]
    southwestFile = data_fold[5]
    northwestFile = data_fold[6]
    abroadFile = data_fold[7]

    vocab = defaultdict(float)
    with open(northeastFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 7,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(northChinaFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 6,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(eastChinaFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 5,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(centralChinaFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 4,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(southChinaFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 3,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(southwestFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 2,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(northwestFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}  # 0 ~ 9
            revs.append(datum)
    with open(abroadFile, "r") as f:
        for line in f:
            rev = []
            rev.append(line.replace(u'\xa0', u'').strip())  # remove the whitespce of the line
            orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_W(word_vecs, k=100):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 100x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "r") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            # print word
            if word.decode("utf-8") in vocab:
                word_vecs[word] = np.fromstring(f.readline(),  sep=' ')
            else:
                f.readline()
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

if __name__ == "__main__":
    w2v_file = "./wiki_chs_pyvec"

    data_fold = ["./northeast", "./northChina", "./eastChina", "./centralChina",
                 "./southChina", "./southwest", "./northwest", "./abroad"]
    print "loading data...",
    revs, vocab = build_data_cv(data_fold, cv=10)
    print "data loaded!"

    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "max sentence length: " + str(max_l)

    print "vocab size: " + str(len(vocab))
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    cPickle.dump([revs, W, word_idx_map, vocab], open('weiboLocation.p', 'w'))
    print "dataset created!"