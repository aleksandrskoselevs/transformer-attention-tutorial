# 1 - Counting Letters

## Description

Consider a sequence, where each element is a randomly selected letter or null/blank. The task is to count how many times each letter appears in the sequence.

For example:

```
Input:
['A'], ['B'], [' '], ['B'], ['C']
Output:
[[1], [2], [1]]
```

The output is also a sequence, where each element corresponds to the count of a letter. In the above case, 'A', 'B' and 'C' appear 1, 2 and 1 times respectively.

We implement the Scaled Dot-Product Attention (described in Section 3.2.1 of [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)) for this task and train without the use of recurrent or convolutional networks.

Here is a sample output from the script.

```
Input: 
[[['A']
  ['A']
  ['A']
  ['B']
  ['C']
  [' ']
  ['A']
  ['B']
  ['B']
  ['A']]]

Prediction: 
[[5 3 1]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [0 1 2 6 9]
[0.154, 0.154, 0.154, 0.042, 0.04, 0.065, 0.154, 0.042, 0.042, 0.154]
Output step 1 attended mainly to Input steps: [3 7 8]
[0.053, 0.053, 0.053, 0.2, 0.05, 0.083, 0.053, 0.2, 0.2, 0.053]
Output step 2 attended mainly to Input steps: [4]
[0.072, 0.072, 0.072, 0.078, 0.286, 0.12, 0.072, 0.078, 0.078, 0.072]
```

For each output step, we see the learned attention being intuitively weighted on the relevant letters. In the above example, Output Step 0 counts the number of 'A's and attended mainly to Input Steps 0, 1, 2, 6 and 9, which were the 'A's in the sequence.

With the `--plot` flag we can also visualize the attention heatmap, which clearly shows the attention focused on the relevant characters for each output.

<div>
<img src="images/attention.png" alt="attention heatmap" width="400px" height="whatever" style="display: block;">
</div>

Just as with a recurrent network, the trained model is able to take in variable sequence lengths, although performance definitely worsens when we deviate from the lengths used in the training set.

## Commands

### Training

```
$ python3 main.py --train
```

This trains a model with default parameters:

- Training steps: `--steps=2000`
- Batchsize: `--batchsize=100`
- Learning rate: `--lr=1e-2`
- Savepath: `--savepath=models/`
- Encoding dimensions: `--hidden=64`

The model will be trained on the Counting Task with default parameters:

- Max sequence length: `--max_len=10`
- Vocabulary size: `--vocab_size=3`

### Testing

```
$ python3 main.py --test
```

This tests the trained model (remember to specify the parameters if the trained model did not use default parameters).

In particular, you can choose whether to plot the attention heatmap (defaults to False):
- Plot: `--plot`

### Help

```
$ python3 main.py --help
```

Run with the `--help` flag to get a list of possible flags.

## Details

Skip ahead to the **Model** section for details about the Attention mechanism.

### Input

The actual input to the model consists of sequences of one-hot embeddings, which means the input tensor has shape `(batchsize, max_len, vocab_size + 1)`. 

*It is `vocab_size + 1` instead of `vocab_size`, because we need to add a one-hot dimension for the null character as well.*

For example, if we set `vocab_size=2`, the possible characters will be 'A', 'B' and ' ' (null). Consider a batch of `batchsize=1` sequence of length `max_len=4`: 

```
[['A', 'B', ' ', 'A']]
```

The embedding for that will be a tensor of shape (1, 4, 3):

```
[[[0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
  [0, 1, 0]]]
```

### Output / Labels

Similar to the input, the output predictions consists of sequences of one-hot embeddings ie. we represent the counts as one-hot vectors. 

The shape of the predictions is determined by the input parameters. First, the length of the prediction corresponds to the `vocab_size`. Next the dimension of each step in a predicted sequence corresponds to `max_len + 1`.

Hence the output/labels tensor should have a shape of `(batchsize, vocab_size, max_len + 1)`.

*It is `max_len + 1` because the minimum count is 0 and the maximum count is `max_len`. So there are `max_len + 1` possible answers.*

If we use the input above as an example, the plain prediction is `[[2, 1]]`, while the embedding is a tensor of shape (1, 2, 5):

```
[[[0, 0, 1, 0, 0],
  [0, 1, 0, 0, 0]]]
```

### Model

**Attention ie. Queries, Keys, Values**

The essence of attention lies in the idea of **Queries**, **Keys** and **Values**. 

Suppose we have a supermarket catalogue of product names and respective prices and I want to calculate the average price of drinks. The **Query** is 'drinks', the **Keys** are all the product names and the **Values** are the respective prices. We focus our attention on **Values** (prices) of **Keys** (product names) that align most with our **Query** ('drinks').

->

The trainnig data will be:

```python
[
  input: [[Pepsi, 0.69$], [Twixies, 0.34$], [Coke, 0.32$]], # product names, prices
  label: [1.01$] # average price of drinkgs
], 
[
  input: [[Snickers, 0.03$], [San Periggglo, 0.21$], [Herring, 0.60$]], # product names, prices
  label: [0.21$] # average price of drinkgs
], 
```

The AI will, based on the training data learn to represent in the latent space 
which products are drinks. It won't be explicitly drinks, but the things it will
be looking for in the latent space align most closely to what we call drinks
in the real world.

---


See how the **Query** and **Keys** have to be the same type? You can see if two words align ('drinks' vs 'water') but you can't align a word with a number, or temperature with prices. Likewise, in the attention mechanism, the **Query** and **Keys** have to have the same dimensions.

In scaled dot-product attention, we calculate alignment using the dot product. Intuitively, we see that if the **Query** and the **Key** are in the same direction, their dot product will be large and positive. If the **Query** and the **Key** are orthogonal, the dot product will be zero. Finally, if the **Query** and the **Key** are in opposite directions, their dot product will be large and negative.

We then allocate more attention on **Values** whose **Keys** align to the **Query**. To do this, we simply multiply the softmax of the dot product by the **Values**. This gives us a weighted sum of the **Values**, where aligned **Keys** contribute more to this weighted sum.

//
Through training we _learn_ to give more attention to the keys that _give_ the correct weighted values

Extract of the `attention` method from the `AttentionModel` class in the script:

```python3
def attention(self, query, key, value):
    # Equation 1 in Vaswani et al. (2017)
    # Scaled dot product between Query and Keys
    scaling_factor = torch.tensor(np.sqrt(query.size()[2]))
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
```

Following the Transformer architecture by Vaswani et al. (2017), the weighted sum is added back to the **Query** via residual connections (Figure 1 and Section 3.1). The sum of **Query** and weighted sum is then layer-normalized and passed to the next step.

Why do we use the sum of the weighted sum and the original **Query** instead of just the weighted sum?

One way of understanding it is because the **Query** may contain important information that cannot be found in the **Values**. To take a simple example, suppose I have two **Key**/**Value** pairs, `[1 0]:[0 1]` and `[0 1]:[0 1]` ie. two different **Keys** `[1 0]` and `[0 1]` with both having the same **Values** `[0 1]` and `[0 1]`. 

If my **Query** is `[1 0]`, then my weighted sum will be `[0 1]` ie. the **Value** of **Key** `[1 0]`. 

On the other hand, if my **Query** is `[0 1]`, my weighted sum will still be `[0 1]` ie. the **Value** of **Key** `[0 1]`. 

If we pass the weighted sum to the next layer, we lose information about the **Query**. The next layer has no way of telling whether my **Query** was `[1 0]` or `[0 1]`. If we instead pass the sum of the weighted sum and the original **Query**, we retain the information since the vector that we pass on can be either `[0 2]` or `[1 1]`.

Now, we need to think about what are our **Queries**, **Keys** and **Values**. 

**Queries**

We want the first step of the output sequence to count the number of 'A's and the second step to count the number of 'B's and so on. Why not have one **Query** vector per output step? The first **Query** vector can check for 'A's and the second **Query** vector can check for 'B's and so on.

In that case, our **Queries** tensor will be of shape `(batch_size, vocab_size, hidden)`. `vocab_size` is the second dimension since we have one **Query** vector for each element in our vocabulary. `hidden` will be the dimension of each **Query** vector.

We will let the model learn the **Queries** tensor, by initializing it as a trainable `nn.Parameter`:

```python3
self.query = nn.Parameter(torch.zeros((1, self.vocab_size, self.hidden)))
```

*Notice the first dimension for `decoder_query` in the code above is 1. This is because we will use `torch.repeat` to change it to `batch_size` at runtime.*

**Keys and Values**

Following the above input/output examples, our input is of shape (1, 4, 3) ie. a length-4 sequence of 3-dim vectors, where each vector is a representation of a character. To create **Key**/**Value** pairs for each character, we can simply pass the input into two regular fully-connected feedforward networks - one for generating a **Key** for each character and one for generating a **Value**.

In fact, we just need one regular fully-connected feedforward network, to generate a **Key**/**Value** for each character ie. the vector acts as both **Key** and **Value**.

Like the **Queries** tensor, we set the output dimension for both networks to `hidden` where `hidden=64` by default. We then end up with a tensor of shape (1, 4, 64). Each 64-dim vector in the tensor acts as both **Key** and **Value** for each character. 

```python3
self.key_val_dense = nn.Linear(self.vocab_size + 1, self.hidden)
```

**Decoding and Softmax**

All that's left is then to compute the scaled dot-product attention, using the **Queries** tensor and the **Keys**/**Values** tensor.

We then pass that to a regular feedforward network to get logits of `self.max_len + 1` dimensions and we just `torch.argmax` these logits to get the predictions.

```python3
decoding, self.attention_weights = self.attention(
    self.query.repeat(key_val.shape[0], 1, 1),
    key_val,
    key_val)
self.final_dense = nn.Linear(self.hidden, self.max_len + 1)
logits = self.final_dense(decoding)
predictions = logits.argmax(dim=2)
```

That's mainly it for the short and simplified demo of attention for the counting task!

Here's a summary figure for the algorithm used in this Task!

<div>
<img src="images/task_1.png" alt="task_1" width="800px" height="whatever" style="display: block;">
</div>

## Notes

If you look at the attention weights printed out when testing the model, you will notice that the same characters are assigned the same weights. That does not seem correct when you consider, for example, language processing - the same words in different positions should be accorded different weights.

In the case of this toy experiment, the order of the input sequence does not matter, since we are just counting characters. But if the order is important, as with the output sequence here, we can use positional encodings, as used by Vaswani et al. (2017) (see Figure 1 and Section 3.5 of the paper).The idea here is to modify the input and output embeddings to help the model differentiate between steps. Vaswani et al. does this by simply adding sine and cosine functions to the embeddings. 

In our case, while position is irrelevant for the input sequence, it is important for the output sequence ie. the meanings of the elements in the output sequence are represented almost entirely by their position eg. we know that the first element in the output counts the number of 'A's by the fact that it is the first element. So we actually do use positional encodings to differentiate between the output steps. Specifically, the **Queries** vector that is initialized as a trainable `nn.Parameter` is our positional encoding for the output. In this case, we allow the model to learn its own positional encoding. 
