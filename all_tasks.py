import string
import numpy as np

import codecs
import json
import regex

class CountingLettersTask(object):
    def __init__(self, max_len=10, vocab_size=3):
        super(CountingLettersTask, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

    def next_batch(self, batchsize=1):
        np.random.seed(69)
        vocab_idx = np.arange(self.vocab_size + 1) # [0, 1, 2, 3]
        batch_max_len_vocab_idx = np.random.choice(vocab_idx, [batchsize, self.max_len]) # [[2, 0, 0, ..., 1, 3, 2]]
        vocab_idx_one_hot_encodings = np.eye(self.vocab_size + 1) # identity matrix
        x = vocab_idx_one_hot_encodings[batch_max_len_vocab_idx] # "replace" the indices with the one-hot encodings
        y = np.eye(self.max_len + 1)[np.sum(x, axis=1)[:, 1:].astype(np.int32)]
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len, self.vocab_size + 1)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        # This means max vocab_size is 26        
        dictionary = np.array(list(' ' + string.ascii_uppercase))
        return dictionary[idx]
    
class TaskDifference(CountingLettersTask):
    def __init__(self, max_len=10, vocab_size=3):
        super().__init__(max_len, vocab_size)
        assert self.vocab_size >= 2, "vocab_size needs to be >= 2 since we need to compute the difference between the first two steps"

    def next_batch(self, batchsize=1):
        np.random.seed(69)
        x = np.eye(self.vocab_size + 1)[np.random.choice(np.arange(self.vocab_size + 1), [batchsize, self.max_len])]
        output = np.sum(x, axis=1)[:, 1:].astype(np.int32)
        diff = np.expand_dims(np.abs(output[:, 0] - output[:, 1]), axis=1)
        output = np.concatenate((output, diff), axis=1)
        y = np.eye(self.max_len + 1)[output]
        return x, y
    
class SignalTask(CountingLettersTask):
    def __init__(self, n_signals=1, seed=696969):
        super().__init__()
        self.n_signals = n_signals # number of letters to count, will increase in task 4
        self.seed = seed

    def next_batch(self, batchsize=1):
        np.random.seed(self.seed)
        signal = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, self.n_signals])]
        seq = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, self.max_len])]
        x = np.concatenate((signal, seq), axis=1)
        # y = np.expand_dims(np.eye(self.max_len + 1)[np.sum(np.argmax(signal, axis=2) == (np.argmax(seq, axis=2)), axis=1)], axis=1)
        y = np.eye(self.max_len + 1)[np.sum(np.expand_dims(np.argmax(signal,axis=2),axis=-1) == np.expand_dims(np.argmax(seq, axis=2), axis=1), axis=2)]
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len + self.n_signals, self.vocab_size)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        # No null character like in the previous tasks
        dictionary = np.array(list(string.ascii_uppercase))
        return dictionary[idx]
    
class TaskThreeSignals(SignalTask):
    def __init__(self):
        super().__init__(n_signals=3, seed=69)


task_dir = '5_translation'

# Translating DE -> EN
class TranslationTask(object):
    def __init__(self):
        self.en_file = f"{task_dir}/data/train.tags.de-en.en"
        self.de_file = f"{task_dir}/data/train.tags.de-en.de"
        self.en_samples = self.get_samples(self.en_file)
        self.de_samples = self.get_samples(self.de_file)
        self.rand_de = np.random.RandomState(1)
        self.rand_en = np.random.RandomState(1)
        self.n_samples = len(self.en_samples)
        self.en_dict = json.load(open(f"{task_dir}/data/en_dict.json", 'r', encoding='utf-8'))
        self.de_dict = json.load(open(f"{task_dir}/data/de_dict.json", 'r', encoding='utf-8'))
        self.en_vocab_size = len(self.en_dict)
        self.de_vocab_size = len(self.de_dict)
        self.idx = 0

    def get_samples(self, file):
        text = codecs.open(file, 'r', 'utf-8').read().lower()
        text = regex.sub("<.*>.*</.*>\r\n", "", text)
        text = regex.sub("[^\n\s\p{Latin}']", "", text)
        samples = text.split('\n')
        return samples

    def embed(self, sample, dictionary, max_len=20, sos=False, eos=False):
        sample = sample.split()[:max_len]
        while len(sample) < max_len:
            sample.append('<PAD>')
        if sos:
            tokens = ['<START>']
        else:
            tokens = []
        tokens.extend(sample)
        if eos:
            tokens.append('<PAD>')
        idxs = []
        for token in tokens:
            try:
                idxs.append(dictionary.index(token))
            except:
                idxs.append(dictionary.index('<UNK>'))
        idxs = np.array(idxs)
        return np.eye(len(dictionary))[idxs]

    def next_batch(self, batchsize=64, max_len=20, idx=None):
        start = self.idx
        if idx is not None:
            start = idx
        end = start + batchsize
        if end > self.n_samples:
            end -= self.n_samples
            en_minibatch_text = self.en_samples[start:]
            self.rand_en.shuffle(self.en_samples)
            en_minibatch_text += self.en_samples[:end]
            de_minibatch_text = self.de_samples[start:]
            self.rand_de.shuffle(self.de_samples)
            de_minibatch_text += self.de_samples[:end]
        else:
            en_minibatch_text = self.en_samples[start:end]
            de_minibatch_text = self.de_samples[start:end]
        self.idx = end
        en_minibatch_in = []
        en_minibatch_out = []
        de_minibatch = []
        for sample in en_minibatch_text:
            # en_minibatch_in: This is the input to the decoder during training.
            # In many models, this is the target sentence (e.g., the corresponding 
            # sentence in English), but shifted one step to the right. This 
            # is often referred to as the "teacher forcing" input, because 
            # it provides the correct output at each step to guide the 
            # training of the decoder and helps the model learn more effectively.                
            en_minibatch_in.append(self.embed(sample, self.en_dict, max_len=max_len, sos=True))
            en_minibatch_out.append(self.embed(sample, self.en_dict, max_len=max_len, eos=True))
        for sample in de_minibatch_text:
            de_minibatch.append(self.embed(sample, self.de_dict, max_len=max_len))
        return np.array(de_minibatch), np.array(en_minibatch_in), np.array(en_minibatch_out)

    def prettify(self, sample, dictionary):
        idxs = np.argmax(sample, axis=1)
        return " ".join(np.array(dictionary)[idxs])