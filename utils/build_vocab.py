import nltk
import pickle
import pandas as pd
from collections import Counter
import CLEF
import sys
reload(sys)
sys.setdefaultencoding('utf8')
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(csv, threshold):
    """Build a simple vocabulary wrapper."""
    clef = CLEF.CLEF()
    ids = clef.ids
    clef_df = clef.clef
    counter = Counter()
    for i, id in enumerate(ids):
        caption = clef_df.loc[id]['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    for split in ['Training']:
        if split == 'Training':
            vocab = build_vocab("./Caption%s.csv", threshold=1)
            with open("./data/caption/vocab.pkl") as f:
                pickle.dump(vocab, f)
                print("Total vocabulary size: %d" % len(vocab))
                print("Saved the vocabulary wrapper to '%s'" % vocab_path)


if __name__ == '__main__':

    main()