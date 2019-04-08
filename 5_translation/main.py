"""
Task 5 - Translation
Demonstration of full Transformer model
"""

import os
import numpy as np
import codecs
import regex
import json
from absl import flags
from absl import app
import seaborn
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("plot", False, "Plot attention heatmaps")
flags.DEFINE_bool("cuda", False, "Use cuda")

# Training parameters
flags.DEFINE_integer("epochs", 1, "Number of training epochs")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 64, "Training batchsize per step")

# Model parameters
flags.DEFINE_integer("heads", 4, "Number of heads for multihead attention")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("dec_layers", 6, "Number of self-attention layers for encodings")
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 20, "Maximum input length from toy task")
flags.DEFINE_integer("line", None, "Line to test")


class Task(object):

    def __init__(self):
        self.en_file = "data/train.tags.de-en.en"
        self.de_file = "data/train.tags.de-en.de"
        self.en_samples = self.get_samples(self.en_file)
        self.de_samples = self.get_samples(self.de_file)
        self.rand_de = np.random.RandomState(1)
        self.rand_en = np.random.RandomState(1)
        self.n_samples = len(self.en_samples)
        self.en_dict = json.load(open("data/en_dict.json", 'r', encoding='utf-8'))
        self.de_dict = json.load(open("data/de_dict.json", 'r', encoding='utf-8'))
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
            en_minibatch_in.append(self.embed(sample, self.en_dict, max_len=max_len, sos=True))
            en_minibatch_out.append(self.embed(sample, self.en_dict, max_len=max_len, eos=True))
        for sample in de_minibatch_text:
            de_minibatch.append(self.embed(sample, self.de_dict, max_len=max_len))
        return np.array(de_minibatch), np.array(en_minibatch_in), np.array(en_minibatch_out)

    def prettify(self, sample, dictionary):
        idxs = np.argmax(sample, axis=1)
        return " ".join(np.array(dictionary)[idxs])


class AttentionModel(nn.Module):

    def __init__(self, en_vocab_size, de_vocab_size,
                 max_len=20, hidden=64,
                 enc_layers=6, dec_layers=6,
                 heads=4):
        super().__init__()

        self.max_len = max_len
        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size
        self.hidden = hidden
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.heads = heads

        self.enc_input_dense = nn.Linear(self.de_vocab_size, self.hidden)
        self.enc_pos_enc = nn.Parameter(torch.zeros((1, self.max_len, self.hidden)))
        self.enc_increase_hidden = nn.ModuleList([
            nn.Linear(self.hidden, self.hidden * 2)
            for i in range(self.enc_layers)])
        self.enc_decrease_hidden = nn.ModuleList([
            nn.Linear(self.hidden * 2, self.hidden)
            for i in range(self.enc_layers)])
        self.enc_layer_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.enc_layers)])
        self.enc_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.enc_layers)])

        self.dec_input_dense = nn.Linear(self.en_vocab_size, self.hidden)
        self.dec_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.hidden)))
        self.dec_increase_hidden = nn.ModuleList([
            nn.Linear(self.hidden, self.hidden * 2)
            for i in range(self.dec_layers)])
        self.dec_decrease_hidden = nn.ModuleList([
            nn.Linear(self.hidden * 2, self.hidden)
            for i in range(self.dec_layers)])
        self.dec_layer_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.dec_layers)])
        self.dec_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.dec_layers)])
        self.enc_dec_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.dec_layers)])
        self.decoder_final_dense = nn.Linear(self.hidden, self.en_vocab_size)

    def forward(self, x1, x2):
        # Embed inputs to hidden dimension
        enc_input_emb = self.enc_input_dense(x1)
        dec_input_emb = self.dec_input_dense(x2)

        # Add positional encodings
        encoding = enc_input_emb + self.enc_pos_enc
        decoding = dec_input_emb + self.dec_pos_enc

        for i in range(self.enc_layers):
            # Encoder Self-Attention
            encoding, _ = self.enc_att[i](
                encoding, encoding, encoding)
            # Encoder dense
            dense = F.relu(self.enc_increase_hidden[i](encoding))
            encoding = encoding + self.enc_decrease_hidden[i](dense)
            encoding = self.enc_layer_norm[i](encoding)

        for i in range(self.dec_layers):
            # Decoder Self-Attention
            decoding, _ = self.dec_att[i](
                decoding, decoding, decoding, mask=True)
            # Encoder-Decoder Attention
            decoding, attention = self.enc_dec_att[i](
                decoding, encoding, encoding)
            # Decoder dense
            dense = F.relu(self.dec_increase_hidden[i](decoding))
            decoding = decoding + self.dec_decrease_hidden[i](dense)
            decoding = self.dec_layer_norm[i](decoding)

        decoding = self.decoder_final_dense(decoding)

        return decoding, attention


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, h, hidden):
        super().__init__()

        self.h = h
        self.hidden = hidden

        self.W_query = nn.Parameter(torch.normal(
            mean=torch.zeros((self.hidden, self.hidden)),
            std=1e-2))
        self.W_key = nn.Parameter(torch.normal(
            mean=torch.zeros((self.hidden, self.hidden)),
            std=1e-2))
        self.W_value = nn.Parameter(torch.normal(
            mean=torch.zeros((self.hidden, self.hidden)),
            std=1e-2))
        self.W_output = nn.Parameter(torch.normal(
            mean=torch.zeros((self.hidden, self.hidden)),
            std=1e-2))

        self.layer_norm = nn.LayerNorm(self.hidden)

    def forward(self, query, key, value, mask=False):
        chunk_size = int(self.hidden/self.h)

        multi_query = torch.matmul(query, self.W_query)\
            .split(split_size=chunk_size, dim=-1)
        multi_query = torch.stack(multi_query, dim=0)

        multi_key = torch.matmul(key, self.W_key)\
            .split(split_size=chunk_size, dim=-1)
        multi_key = torch.stack(multi_key, dim=0)

        multi_value = torch.matmul(value, self.W_value)\
            .split(split_size=chunk_size, dim=-1)
        multi_value = torch.stack(multi_value, dim=0)

        scaling_factor = torch.tensor(np.sqrt(multi_query.shape[-1]))
        dotp = torch.matmul(multi_query, multi_key.transpose(2, 3)) / scaling_factor
        attention_weights = F.softmax(dotp, dim=-1)

        if mask:
            attention_weights = attention_weights.tril()
            attention_weights = attention_weights / attention_weights.sum(dim=3, keepdim=True)

        weighted_sum = torch.matmul(attention_weights, multi_value)
        weighted_sum = weighted_sum.split(1, dim=0)
        weighted_sum = torch.cat(weighted_sum, dim=-1).squeeze()

        output = weighted_sum + query
        output = self.layer_norm(output)
        return output, attention_weights


class TaskDataset(Dataset):
    def __init__(self, task, max_len):
        super().__init__()
        self.task = task
        self.max_len = max_len

    def __len__(self):
        return self.task.n_samples

    def __getitem__(self, i):
        ans = [x[0] for x in self.task.next_batch(
            batchsize=1,
            max_len=self.max_len,
            idx=i)]
        return ans


def train(max_len=20,
          hidden=64,
          enc_layers=1,
          dec_layers=6,
          heads=4,
          batchsize=64,
          epochs=1,
          print_every=50,
          savepath='models/',
          cuda=torch.cuda.is_available()):

    os.makedirs(savepath, exist_ok=True)
    task = Task()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    model = AttentionModel(task.en_vocab_size,
                           task.de_vocab_size,
                           max_len,
                           hidden,
                           enc_layers,
                           dec_layers,
                           heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    task_dataset = TaskDataset(task, int(max_len))
    task_dataloader = DataLoader(task_dataset,
                                 batch_size=batchsize,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=7)

    for i in range(epochs):
        print('Epoch: ', i)
        this_epoch_loss = 0
        for j, a_batch in enumerate(task_dataloader):
            minibatch_enc_in, minibatch_dec_in, minibatch_dec_out = a_batch
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                minibatch_enc_in = minibatch_enc_in.float().to(device)
                minibatch_dec_in = minibatch_dec_in.float().to(device)
                minibatch_dec_out = minibatch_dec_out.to(device)
                out, _ = model(minibatch_enc_in, minibatch_dec_in)
                loss = F.cross_entropy(
                    out.transpose(1, 2),
                    minibatch_dec_out.argmax(dim=2))
                loss.backward()
                optimizer.step()
            loss = loss.detach().cpu().numpy()
            this_epoch_loss += loss
            if (j + 1) % print_every == 0:
                print("Iteration {} - Loss {}".format(j + 1, loss))

        this_epoch_loss /= (j + 1)
        print("Epoch {} - Loss {}".format(i, this_epoch_loss))

        lr_scheduler.step(this_epoch_loss)

        torch.save(model.state_dict(), savepath + '/ckpt_{}.pt'.format(str(i)))
        print('Model saved')

    print("Training complete!")
    torch.save(model.state_dict(), savepath + '/ckpt.pt')


def test(max_len=20,
         hidden=64,
         enc_layers=1,
         dec_layers=6,
         heads=4,
         savepath='models/',
         plot=True,
         line=198405,
         cuda=torch.cuda.is_available()):

    task = Task()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    model = AttentionModel(task.en_vocab_size,
                           task.de_vocab_size,
                           max_len,
                           hidden,
                           enc_layers,
                           dec_layers,
                           heads).to(device)
    model.load_state_dict(torch.load(savepath + '/ckpt.pt'))

    idx = line
    if idx is None:
        idx = np.random.randint(low=0, high=task.n_samples)
        print('Predicting line :', idx)

    samples, _, truth = task.next_batch(batchsize=1, max_len=max_len, idx=idx)
    print("\nInput : \n{}".format(regex.sub("\s<PAD>", "", task.prettify(samples[0], task.de_dict))))
    print("\nTruth : \n{}".format(regex.sub("\s<PAD>", "", task.prettify(truth[0], task.en_dict))))

    output = ""
    for i in range(max_len):
        predictions, attention = model(
            torch.Tensor(samples).to(device),
            torch.Tensor(task.embed(output, task.en_dict, sos=True)).to(device))
        predictions = predictions.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()
        output += " " + task.prettify(predictions[0], task.en_dict).split()[i]
    print("\nOutput: \n{}".format(regex.sub("\s<PAD>", "", task.prettify(predictions[0], task.en_dict))))

    if plot:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.5)
        x_labels = regex.sub("\s<PAD>", "", task.prettify(samples[0], task.de_dict)).split()
        y_labels = regex.sub("\s<PAD>", "", task.prettify(predictions[0], task.en_dict)).split()
        for i in range(4):
            ax = plt.Subplot(fig, gs[i])
            seaborn.heatmap(
                data=attention[:, 0, :, :][i, :len(y_labels), :len(x_labels)],
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                cmap='plasma',
                vmin=np.min(attention),
                vmax=np.max(attention),
                cbar=False)
            ax.set_title("Head {}".format(i))
            ax.set_aspect('equal')
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
            fig.add_subplot(ax)
        plt.show()


def main(unused_args):
    if FLAGS.train:
        train(FLAGS.max_len, FLAGS.hidden,
              FLAGS.enc_layers, FLAGS.dec_layers,
              FLAGS.heads, FLAGS.batchsize,
              FLAGS.epochs, FLAGS.print_every,
              FLAGS.savepath, FLAGS.cuda)
    elif FLAGS.test:
        test(FLAGS.max_len, FLAGS.hidden,
             FLAGS.enc_layers, FLAGS.dec_layers,
             FLAGS.heads, FLAGS.savepath,
             FLAGS.plot, FLAGS.line, FLAGS.cuda)
    else:
        print('Specify train or test option')


if __name__ == "__main__":
    app.run(main)
