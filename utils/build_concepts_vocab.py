import nltk
import pickle
import pandas as pd
from collections import Counter
import CLEF
import sys
reload(sys)
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        # use special label to indicate padded concept.
        self.concept2idx = {}

        self.idx2concept = {}
        self.concept2means = {}
    def get_means(self, concept):
        pass

    def __call__(self, concept):
        return self.concept2idx[concept]
    def getConcept(self, idx):
        return self.idx2concept[idx]
    def __len__(self):
        return len(self.concept2idx)

    def build_vocab(self, c2i_path, i2c_path):
        """Build a simple vocabulary wrapper."""
        with open(c2i_path, 'r') as c2i:
            self.concept2idx = pickle.load(c2i)
        with open(i2c_path, 'r') as i2c:
            self.idx2concept = pickle.load(i2c)


def main():
    voc = Vocabulary()
    voc.build_vocab('concept2idx_sample.pkl', 'idx2concept_sample.pkl')
    with open('concept_voc_sample.pkl', 'wb') as f:
        pickle.dump(voc, f)
if __name__ == '__main__':

    main()