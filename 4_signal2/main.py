"""
Task 4 - Signal 2
Demonstration of Multi-head Attention
"""

import os
import string
import numpy as np
from absl import flags
from absl import app

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training parameters
flags.DEFINE_integer("steps", 4000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")

# Model parameters
flags.DEFINE_bool("multihead", True, "Whether to use multihead or singlehead attention")
flags.DEFINE_integer("heads", 4, "Number of heads for multihead attention")
flags.DEFINE_bool("pos_enc", True, "Whether to use positional encodings")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
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
        # np.random.seed(69)
        if signal is not None:
            signal = string.ascii_uppercase.index(signal)
            signal = np.eye(self.vocab_size)[np.ones((batchsize, 1), dtype=int) * signal]
        else:
            signal = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, 3])]
        seq = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, self.max_len])]
        x = np.concatenate((signal, seq), axis=1)
        y = np.eye(self.max_len + 1)[np.sum(np.expand_dims(np.argmax(signal,axis=2),axis=-1) == np.expand_dims(np.argmax(seq, axis=2), axis=1), axis=2)]
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len + 3, self.vocab_size)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        dictionary = np.array(list(string.ascii_uppercase))
        return dictionary[idx]


class AttentionModel(nn.Module):
    def __init__(self, max_len=10, vocab_size=3, hidden=64,
                 pos_enc=True, enc_layers=1,
                 use_multihead=True, heads=4):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.pos_enc = pos_enc
        self.enc_layers = enc_layers
        self.use_multihead = use_multihead
        self.heads = heads

        self.att_layer_norm = nn.LayerNorm(self.hidden)
        self.input_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 3, self.hidden)))

        self.base_enc_layer = nn.Linear(self.vocab_size, self.hidden)
        self.enc_layer_att_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.enc_layers)])
        self.enc_increase_hidden = nn.ModuleList([
            nn.Linear(self.hidden, self.hidden * 2)
            for i in range(self.enc_layers)])
        self.enc_decrease_hidden = nn.ModuleList([
            nn.Linear(self.hidden * 2, self.hidden)
            for i in range(self.enc_layers)])
        self.enc_layer_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.enc_layers)])

        self.decoder_input = nn.Parameter(torch.zeros((1, 3, self.hidden)))
        self.decoder_att_norm = nn.LayerNorm(self.hidden)
        self.dec_increase_hidden = nn.Linear(self.hidden, self.hidden * 2)
        self.dec_decrease_hidden = nn.Linear(2 * self.hidden, self.hidden)
        self.dec_layer_norm = nn.LayerNorm(self.hidden)

        self.decoder_dense = nn.Linear(self.hidden, self.max_len + 1)

        if use_multihead:
            self.enc_mult_att = nn.ModuleList([
                MultiHeadAttention(self.heads, self.hidden)
                for i in range(self.enc_layers)
            ])      
            self.enc_dec_mult_att = MultiHeadAttention(self.heads, self.hidden)

    def forward(self, x): # (13, 3)
        encoding = self.base_enc_layer(x) # (13, 64)

        if self.pos_enc:
            encoding = encoding + self.input_pos_enc

        for i in range(self.enc_layers):
            qkv = encoding
            if self.use_multihead:
                encoding, _ = self.enc_mult_att[i](encoding, encoding, encoding)
                
                add = encoding + qkv
                add_and_norm = self.enc_layer_att_norm[i](add)
                encoding = add_and_norm
            else:
                encoding, _ = self.attention(encoding, encoding, encoding)

            dense = F.relu(self.enc_increase_hidden[i](encoding)) # Feed Forward
            encoding = encoding + self.enc_decrease_hidden[i](dense)
            encoding = self.enc_layer_norm[i](encoding)

        query = self.decoder_input.repeat(x.shape[0], 1, 1)
        if self.use_multihead:            
            decoding, attention_weights = self.enc_dec_mult_att(
                self.decoder_input.repeat(x.shape[0], 1, 1),
                encoding,
                encoding
            )
            
            add = decoding + query
            add_and_norm = self.decoder_att_norm(add)
            decoding = add_and_norm              
        else:
            decoding, attention_weights = self.attention(
                query,
                encoding,
                encoding)
            
        query = decoding
        decoding = self.dec_decrease_hidden(F.relu(self.dec_increase_hidden(decoding))) # Feed Forward
        add = query + decoding
        add_and_norm = self.dec_layer_norm(add)
        decoding = self.decoder_dense(add_and_norm)

        return decoding, attention_weights, self.input_pos_enc

    def attention(self, query, key, value):
        # Equation 1 in Vaswani et al. (2017)
        # Scaled dot product between Query and Keys
        scaling_factor = torch.tensor(np.sqrt(query.shape[-1]))
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
        output = self.att_layer_norm(output)

        return output, attention_weights

def scaled_dot_product_attention(query, key, value):
    d_k = query.shape[-1] # 64
    scaling_factor = torch.tensor(np.sqrt(d_k)) # 8
    dot_product = torch.bmm(
        query, key.transpose(1, 2) # (1, 3, 64) x (1, 64, 10) = (1, 3, 10)
    ) 
    scaled_dot_product = dot_product / scaling_factor

    attention_weights = F.softmax(scaled_dot_product, dim=2)
    weighted_sum = torch.bmm(attention_weights, value) # (1, 3, 10) x (1, 10, 64) = (1, 3, 64)

    return weighted_sum, attention_weights 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=6, hidden=64):
        super().__init__()
        self.num_heads = num_heads
        self.hidden = hidden

        # W as in Weight tensor
        self.W_q = nn.Linear(hidden, hidden)
        self.W_k = nn.Linear(hidden, hidden)
        self.W_v = nn.Linear(hidden, hidden)
        self.W_output = nn.Linear(hidden, hidden)

        self.d_v = int(hidden / num_heads) # == d_q == d_k == 64 / 4 = 16

    def forward(self, query, key, value): # encoding: (13, 64), decoding: query=(3, 64)           
        heads = [None] * self.num_heads # create an empty array of size num_heads
        attention_weights = [None] * self.num_heads # for visualization
        
        W_query_projected = self.W_q(query) # (13, 64)
        W_query_split = W_query_projected.split(split_size=self.d_v, dim=-1) # (4, (batch_size, 13, 16))
        W_key_projected = self.W_k(key)
        W_key_split = W_key_projected.split(split_size=self.d_v, dim=-1)
        W_value_projected = self.W_v(value)
        W_value_split = W_value_projected.split(split_size=self.d_v, dim=-1)

        for i in range(self.num_heads):
            weighted_sum, attention_weight  = scaled_dot_product_attention(
                W_query_split[i], # (13, 16)
                W_key_split[i],
                W_value_split[i],
            ) # page 5 of Vaswani et al. (2017)

            heads[i] = weighted_sum # (13, 16)
            attention_weights[i] = attention_weight # for visualization
        
        concat = torch.cat(heads, dim=-1) # (13, 64)
        linear = self.W_output(concat) # (13, 64)

        output = linear

        return output, attention_weights


def train(max_len=10,
          vocab_size=3,
          hidden=64,
          pos_enc=True,
          enc_layers=2,
          use_multihead=True,
          heads=4,
          batchsize=100,
          steps=4000,
          print_every=50,
          savepath='models/'):

    os.makedirs(savepath, exist_ok=True)
    model = AttentionModel(max_len=max_len,
                           vocab_size=vocab_size,
                           hidden=hidden,
                           pos_enc=pos_enc,
                           enc_layers=enc_layers,
                           use_multihead=use_multihead,
                           heads=heads)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)
    task = Task(max_len=max_len, vocab_size=vocab_size)

    loss_hist = []
    for i in range(steps):
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
            lr_scheduler.step(loss)
        if (i + 1) % print_every == 0:
            print("Iteration {} - Loss {}".format(i + 1, loss))
        loss_hist.append(loss.detach().numpy())

    print("Iteration {} - Loss {}".format(i + 1, loss))
    print("Training complete!")
    torch.save(model.state_dict(), savepath + '/ckpt.pt')
    return loss_hist


def test(max_len=10,
         vocab_size=3,
         hidden=64,
         pos_enc=True,
         enc_layers=2,
         use_multihead=True,
         heads=4,
         savepath='models/',
         plot=True):

    model = AttentionModel(max_len=max_len, vocab_size=vocab_size,
                           hidden=hidden, pos_enc=pos_enc,
                           enc_layers=enc_layers,
                           use_multihead=use_multihead,
                           heads=heads)
    model.load_state_dict(torch.load(savepath + '/ckpt.pt'))
    task = Task(max_len=max_len, vocab_size=vocab_size)

    samples, labels = task.next_batch(batchsize=1)
    print("\nInput: \n{}".format(task.prettify(samples)))
    model.eval()
    with torch.set_grad_enabled(False):
        predictions, attention, input_pos_enc = model(torch.Tensor(samples))
    predictions = predictions.detach().numpy()
    predictions = predictions.argmax(axis=2)

    print("\nPrediction: \n{}".format(predictions))
    print("\nEncoder-Decoder Attention: ")
    if use_multihead:
        for h, head in enumerate(attention[0]):
            print("Head #{}".format(h))
            for i, output_step in enumerate(head):
                print("\tAttention of Output step {} to Input steps".format(i))
                print("\t{}".format([float("{:.3f}".format(step)) for step in output_step]))
    else:
        for i, output_step in enumerate(attention[0]):
            print("Output step {} attended mainly to Input steps: {}".format(i, np.where(output_step >= np.max(output_step))[0]))
            print([float("{:.3f}".format(step)) for step in output_step])
    if pos_enc:
        input_pos_enc = input_pos_enc.detach().numpy()
        print("\nL2-Norm of Input Positional Encoding:")
        print([float("{:.3f}".format(step)) for step in np.linalg.norm(input_pos_enc, ord=2, axis=2)[0]])


def main(unused_args):
    if FLAGS.train:
        train(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
              FLAGS.pos_enc, FLAGS.enc_layers,
              FLAGS.multihead, FLAGS.heads,
              FLAGS.batchsize, FLAGS.steps, FLAGS.print_every,
              FLAGS.savepath)
    elif FLAGS.test:
        test(FLAGS.max_len, FLAGS.vocab_size, FLAGS.hidden,
             FLAGS.pos_enc, FLAGS.enc_layers, FLAGS.savepath,
             FLAGS.plot)
    else:
        print('Specify train or test option')


if __name__ == "__main__":
    app.run(main)
