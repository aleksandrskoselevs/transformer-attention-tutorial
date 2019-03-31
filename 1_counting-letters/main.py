"""
Task 1 - Counting Letters
Simple implementation of attention and tutorial on queries, keys and values
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
flags.DEFINE_integer("steps", 2000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")
flags.DEFINE_float("lr", 1e-2, "Learning rate")

# Model parameters
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 10, "Maximum input length from toy task")
flags.DEFINE_integer("vocab_size", 3, "Size of input vocabulary, not including ' ' (null character)")


class Task(object):

    def __init__(self, max_len=10, vocab_size=3):
        super(Task, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

    def next_batch(self, batchsize=100):
        x = np.eye(self.vocab_size + 1)[np.random.choice(np.arange(self.vocab_size + 1), [batchsize, self.max_len])]
        y = np.eye(self.max_len + 1)[np.sum(x, axis=1)[:, 1:].astype(np.int32)]
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len, self.vocab_size + 1)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        dictionary = np.array(list(' ' + string.ascii_uppercase))
        return dictionary[idx]


class AttentionModel(nn.Module):
    def __init__(self, max_len=10, vocab_size=3, hidden=64):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden = hidden

        self.query = nn.Parameter(
            torch.zeros((1, self.vocab_size, self.hidden),
                        requires_grad=True))
        self.key_val_dense = nn.Linear(self.vocab_size + 1, self.hidden)
        self.layer_norm = nn.LayerNorm(self.hidden)
        self.final_dense = nn.Linear(self.hidden, self.max_len + 1)

    def forward(self, x):
        key_val = self.key_val_dense(x)
        decoding, self.attention_weights = self.attention(
            self.query.repeat(key_val.shape[0], 1, 1),
            key_val,
            key_val)
        logits = self.final_dense(decoding)
        return logits, self.attention_weights

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
          batchsize=100,
          steps=2000,
          print_every=50,
          savepath='models/'):

    os.makedirs(savepath, exist_ok=True)
    model = AttentionModel(max_len=max_len, vocab_size=vocab_size, hidden=hidden)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=250, verbose=True)
    task = Task(max_len=max_len, vocab_size=vocab_size)

    loss_hist = []
    for i in range(steps):
        minibatch_x, minibatch_y = task.next_batch(batchsize=batchsize)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            minibatch_x = torch.Tensor(minibatch_x)
            minibatch_y = torch.Tensor(minibatch_y)
            out, _ = model(minibatch_x)
            loss = F.cross_entropy(
                out.transpose(1, 2),
                minibatch_y.argmax(dim=2))
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
        if (i + 1) % print_every == 0:
            print("Iteration {} - Loss {}".format(i + 1, loss))
        loss_hist.append(loss.detach().numpy())

    print("Iteration {} - Loss {}".format(i + 1, loss))
    print("Training complete!")
    torch.save(model.state_dict(), savepath + '/ckpt.pt')
    return np.array(loss_hist)


def test(max_len=10,
         vocab_size=3,
         hidden=64,
         savepath='models/',
         plot=True):

    model = AttentionModel(max_len=max_len, vocab_size=vocab_size, hidden=hidden)
    model.load_state_dict(torch.load(savepath + '/ckpt.pt'))
    task = Task(max_len=max_len, vocab_size=vocab_size)

    samples, labels = task.next_batch(batchsize=1)
    print("\nInput: \n{}".format(task.prettify(samples)))
    model.eval()
    with torch.set_grad_enabled(False):
        predictions, attention = model(torch.Tensor(samples))
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
            yticklabels=["output_0", "output_1", "output_2"],
            xticklabels=task.prettify(samples).reshape(-1),
            ax=ax,
            cmap='plasma',
            cbar=True,
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_aspect('equal')
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
        plt.show()
    return samples, labels, predictions, attention


def main(unused_args):
    if FLAGS.train:
        train(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
              FLAGS.batchsize, FLAGS.steps, FLAGS.print_every,
              FLAGS.savepath)
    elif FLAGS.test:
        test(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
             FLAGS.savepath, FLAGS.plot)
    else:
        print('Specify train or test option')


if __name__ == "__main__":
    app.run(main)
