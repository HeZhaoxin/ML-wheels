from gensim import corpora
import collections
from gensim. .models.word2vec import Word2Vec

sentences = [['A1'，'A2']，[]，[]，....]
model = Word2Vec()
model.build_vocab(sentences)
model.train(sentences，total_examples = model.corpus_count，epochs = model.iter)

def flatten(lst):
    for item in lst:
        if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


class word2vec:
    def __init__(
            self,
            sentences,
            size=256,
            window=5,
            min_count=5,
            sg=1,
            hs=1,
            sample=1e-3,
            seed=1,
            iter=5):
        self.vocab = {}
        self.vocab_size = 0
        self.train(sentences)

    def _extend_vocab(self, lst):
        for w in lst:
            self.vocab[w] = self.vocab_size
            self.vocab_size += 1

    def train(self, sentences):
        self._extend_vocab(flatten(sentences))
        self.embed_init()
        self.output_weight_init()
        train_data = self.get_data(sentences)
