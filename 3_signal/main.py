"""
Task 3 - Signal
Demonstration of positional encodings
"""

import os
import string
import numpy as np
from absl import flags
from absl import app
import seaborn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("plot", False, "Plot attention heatmap during testing")

# Training parameters
flags.DEFINE_integer("steps", 1000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")

# Model parameters
flags.DEFINE_bool("pos_enc", False, "Whether to use positional encodings")
flags.DEFINE_integer("num_enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 10, "Maximum input length from toy task")
flags.DEFINE_integer("vocab_size", 3, "Size of input vocabulary")


class Task(object):

    def __init__(self, max_len=10, vocab_size=3):
        super(Task, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

    def next_batch(self, batchsize=100, signal=None):
        if signal is not None:
            signal = string.ascii_uppercase.index(signal)
            signal = np.eye(self.vocab_size)[np.ones((batchsize, 1), dtype=int) * signal]
        else:
            signal = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, 1])]
        seq = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, self.max_len])]
        x = np.concatenate((signal, seq), axis=1)
        y = np.expand_dims(np.eye(self.max_len + 1)[np.sum(np.argmax(signal, axis=2) == (np.argmax(seq, axis=2)), axis=1)], axis=1)
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len + 1, self.vocab_size)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        # This means max vocab_size is 26
        dictionary = np.array(list(string.ascii_uppercase))
        return dictionary[idx]


class AttentionModel(nn.Module):
    def __init__(self, max_len=10, vocab_size=3, hidden=64,
                 pos_enc=True, num_enc_layers=1):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.pos_enc = pos_enc
        self.num_enc_layers = num_enc_layers

        self.layer_norm = nn.LayerNorm(self.hidden)
        self.input_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.hidden)))

        self.base_enc_layer = nn.Linear(self.vocab_size, self.hidden)
        self.enc_layers = []
        for i in range(self.num_enc_layers):
            self.enc_layers.append([
                nn.Linear(self.hidden, self.hidden * 2),
                nn.Linear(self.hidden * 2, self.hidden)])

        self.decoder_input = nn.Parameter(torch.zeros((1, 1, self.hidden)))
        self.decoder_dense = nn.Linear(self.hidden, self.max_len + 1)

    def forward(self, x):
        encoding = self.base_enc_layer(x)

        if self.pos_enc:
            # Add positional encodings
            encoding += self.input_pos_enc

        for i in range(self.num_enc_layers):
            encoding, _ = self.attention(encoding, encoding, encoding)
            dense = F.relu(self.enc_layers[i][0](encoding))
            encoding = encoding + self.enc_layers[i][1](dense)
            encoding = self.layer_norm(encoding)

        decoding, attention_weights = self.attention(
            self.decoder_input.repeat(x.shape[0], 1, 1),
            encoding,
            encoding)
        decoding = self.decoder_dense(decoding)

        return decoding, attention_weights, self.input_pos_enc

    def attention(self, query, key, value):
        # Equation 1 in Vaswani et al. (2017)
        # Scaled dot product between Query and Keys
        scaling_factor = torch.tensor(np.sqrt(query.shape[2]))
        output = torch.bmm(
            query, key.transpose(1, 2)
        ) / scaling_factor

        # Softmax to get attention weights
        attention_weights = F.softmax(output, dim=2)

        # Multiply weights by Values
        weighted_sum = torch.bmm(attention_weights, value)

        # Following Figure 1 and Section 3.1 in Vaswani et al. (2017)
        # Residual connection ie. add weighted sum to original query
        output = weighted_sum + query

        # Layer normalization
        output = self.layer_norm(output)

        return output, attention_weights


def train(max_len=10,
          vocab_size=3,
          hidden=64,
          pos_enc=False,
          num_enc_layers=1,
          batchsize=100,
          steps=2000,
          print_every=50,
          savepath='models/'):

    os.makedirs(savepath, exist_ok=True)
    model = AttentionModel(max_len=max_len, vocab_size=vocab_size,
                           hidden=hidden, pos_enc=pos_enc,
                           num_enc_layers=num_enc_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    task = Task(max_len=max_len, vocab_size=vocab_size)

    for i in np.arange(steps):
        minibatch_x, minibatch_y = task.next_batch(batchsize=batchsize)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            minibatch_x = torch.Tensor(minibatch_x)
            minibatch_y = torch.Tensor(minibatch_y)
            out, _, _ = model(minibatch_x)
            loss = F.cross_entropy(
                out.transpose(1, 2),
                minibatch_y.argmax(dim=2))
            loss.backward()
            optimizer.step()
        if (i + 1) % print_every == 0:
            print("Iteration {} - Loss {}".format(i + 1, loss))

    print("Iteration {} - Loss {}".format(i + 1, loss))
    print("Training complete!")
    torch.save(model.state_dict(), savepath + '/ckpt.pt')


def test(max_len=10,
         vocab_size=3,
         hidden=64,
         pos_enc=False,
         num_enc_layers=1,
         savepath='models/',
         plot=True):

    model = AttentionModel(max_len=max_len, vocab_size=vocab_size,
                           hidden=hidden, pos_enc=pos_enc,
                           num_enc_layers=num_enc_layers)
    model.load_state_dict(torch.load(savepath + '/ckpt.pt'))
    task = Task(max_len=max_len, vocab_size=vocab_size)

    samples, labels = task.next_batch(batchsize=1)
    print("\nInput: \n{}".format(task.prettify(samples)))
    model.eval()
    with torch.set_grad_enabled(False):
        predictions, attention, input_pos_enc = model(torch.Tensor(samples))
    predictions = predictions.detach().numpy()
    predictions = predictions.argmax(axis=2)
    attention = attention.detach().numpy()

    print("\nPrediction: \n{}".format(predictions))
    print("\nEncoder-Decoder Attention: ")
    for i, output_step in enumerate(attention[0]):
        print("Output step {} attended mainly to Input steps: {}".format(
            i, np.where(output_step >= np.max(output_step))[0]))
        print([float("{:.3f}".format(step)) for step in output_step])

    if plot:
        fig, ax = plt.subplots()
        seaborn.heatmap(
            attention[0],
            yticklabels=["output_0"],
            xticklabels=task.prettify(samples).reshape(-1),
            ax=ax,
            cmap='plasma',
            cbar=True,
            cbar_kws={"orientation": "horizontal"})
        ax.set_aspect('equal')
        ax.set_title("Encoder-Decoder Attention")
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
        plt.show()

    if pos_enc:
        input_pos_enc = input_pos_enc.detach().numpy()
        print("\nL2-Norm of Input Positional Encoding:")
        print([
            float("{:.3f}".format(step))
            for step in np.linalg.norm(input_pos_enc, ord=2, axis=2)[0]])

        if plot:
            fig2, ax2 = plt.subplots()
            seaborn.heatmap(
                [np.linalg.norm(input_pos_enc, ord=2, axis=2)[0]],
                vmin=0.,
                # vmax=.5,
                yticklabels=["L2-Norm"],
                xticklabels=task.prettify(samples).reshape(-1),
                ax=ax2,
                cmap='plasma',
                cbar=True,
                cbar_kws={"orientation": "horizontal"})
        ax2.set_aspect('equal')
        ax2.set_title("Positional Encodings L2-Norm")
        for tick in ax2.get_yticklabels():
            tick.set_rotation(0)
        plt.show()


def main(unused_args):
    if FLAGS.train:
        train(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
              FLAGS.pos_enc, FLAGS.num_enc_layers, FLAGS.batchsize,
              FLAGS.steps, FLAGS.print_every, FLAGS.savepath)
    elif FLAGS.test:
        test(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
             FLAGS.pos_enc, FLAGS.num_enc_layers, FLAGS.savepath,
             FLAGS.plot)
    else:
        print('Specify train or test option')


if __name__ == "__main__":
    app.run(main)
