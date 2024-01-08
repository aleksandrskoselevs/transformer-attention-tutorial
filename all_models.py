import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=False):
    d_k = query.shape[2] # 64
    scaling_factor = torch.tensor(np.sqrt(d_k)) # 8
    dot_product = torch.bmm(
        query, key.transpose(1, 2) # (1, 3, 64) x (1, 64, 10) = (1, 3, 10)
    ) 
    scaled_dot_product = dot_product / scaling_factor
    
    # `mask` explained in Task 5
    if mask:
        mask = torch.triu(torch.ones_like(scaled_dot_product), diagonal=1).bool()
        scaled_dot_product = scaled_dot_product.masked_fill_(mask, -np.inf)

    attention_weights = F.softmax(scaled_dot_product, dim=2)
    weighted_sum = torch.bmm(attention_weights, value) # (1, 3, 10) x (1, 10, 64) = (1, 3, 64)

    return weighted_sum, attention_weights

class AttentionModel(nn.Module):
    def __init__(self, max_len=10, vocab_size=3, hidden=64):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden = hidden

        self.query = nn.Parameter(torch.zeros((1, self.vocab_size, self.hidden), requires_grad=True))
        self.key_val_dense = nn.Linear(self.vocab_size + 1, self.hidden)
        
        self.norm = nn.LayerNorm(self.hidden)
        self.linear = nn.Linear(self.hidden, self.max_len + 1) # final output

    def forward(self, x): # x: (1, 10, 4)
        key_val = self.key_val_dense(x) # (1, 10, 64)
        query = self.query.repeat(key_val.shape[0], 1, 1) # (1, 3, 64)

        attention, self.attention_weights = scaled_dot_product_attention(  # (1, 3, 64), (1, 3, 10)
            query,
            key_val,
            key_val)
        
        add_and_norm = self.norm(attention + query) # (1, 3, 64)        
        linear = self.linear(add_and_norm) # (1, 3, 11)
        
        return linear, self.attention_weights
    
class SelfAttentionModel(AttentionModel):
    def __init__(self, self_att=False):
        super().__init__()
        
        self.self_att = self_att
        # +1 extra output for difference
        self.query = nn.Parameter(torch.zeros((1, self.vocab_size + 1, self.hidden), requires_grad=True))

    def forward(self, x):
        key_val = self.key_val_dense(x)
        query = self.query.repeat(key_val.shape[0], 1, 1) # (1, 3, 64)

        attention, enc_attention_weights = scaled_dot_product_attention(  # (1, 3, 64), (1, 3, 10)
            query,
            key_val,
            key_val)
        
        add_and_norm = self.norm(attention + query) # (1, 3, 64)
                
        # Begin: Difference from AttentionModel
        decoding_pre = add_and_norm
        # This can be thought of N = 2 layers of attention, i.e. the Nx in the
        # diagram
        if self.self_att:
            attention, self_attention_weights = scaled_dot_product_attention(  # (1, 3, 64), (1, 3, 10)
                decoding_pre,
                decoding_pre,
                decoding_pre)
            
            add_and_norm = self.norm(attention + query) # (1, 3, 64)

            decoding = add_and_norm # decoding_post
        else:
            self_attention_weights = None
            decoding = decoding_pre
        # End: Difference from AttentionModel
        
        linear = self.linear(decoding)
        
        return linear, enc_attention_weights, self_attention_weights
    
class PosEncLayersAttentionModel(SelfAttentionModel):
    def __init__(self, pos_enc=True, num_enc_layers=1):
        super().__init__()

        self.query = None # we will have many queries, keys and values now
        self.key_val_dense = None 

        self.pos_enc = pos_enc
        self.num_enc_layers = num_enc_layers
        
        self.input_embedding = nn.Linear(self.vocab_size, self.hidden) # can be
        # thought of as the query from previous exercise
        self.input_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.hidden)))
        
        # `enc_` as in Encoder, the left side of the diagram       
        # Creates an array of num_enc_layers nn.Sequential objects. Each is part
        # of its own layer, and does not share weights with the
        self.enc_feed_forwad = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden, self.hidden * 2), # (11, 128)
                nn.ReLU(),
                nn.Linear(self.hidden * 2, self.hidden) # (11, 64)
            )
            for i in range(self.num_enc_layers)
        ])
        self.enc_layer_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.num_enc_layers)])

        self.output_embedding = nn.Linear(1, self.hidden)

    def forward(self, x): # x: (11, 3)
        encoding = self.input_embedding(x) # (11, 64)
        
        if self.pos_enc:
            # Add positional encodings
            encoding += self.input_pos_enc # (11, 64)

        #  After each iteration of the loop, encoding remains a tensor of shape 
        # (100, 11, 64). Our output becomes the input for the next layer.
        for i in range(self.num_enc_layers):
            attention, _ = scaled_dot_product_attention(
                encoding, 
                encoding, 
                encoding
            )   
            # This was implemented differently than the diagram. Each layer 
            # should have its own norm also in this part. 
            add_and_norm = self.norm(attention + encoding)

            # We have #num_enc_layers "separate networks", each trained for
            # a specific layer.
            feed_forward = self.enc_feed_forwad[i](add_and_norm) # (11, 128) -> (11, 64)
            add_and_norm = self.enc_layer_norm[i](feed_forward + add_and_norm)

            encoding = add_and_norm # (11, 64)

        # Right side
        # We know that we expect the output to be 1 token, 
        # so we can hardcode it here.
        # This can be done with nn.Parameter
        ones = torch.ones(x.shape[0], 1, 1)
        output_embedding = self.output_embedding(ones) # (1, 64)
        
        attention, attention_weights = scaled_dot_product_attention( # (1, 64), (1, 11)
            output_embedding, 
            encoding,  # (11, 64)
            encoding
        )
        add_and_norm = self.norm(attention + output_embedding)
            
        linear = self.linear(add_and_norm) # (1, 11)

        return linear, attention_weights, self.input_pos_enc

# We implement use only multi-headed attention here, without an option to switch
# back to regular attention. 
# The task folder provides an implmentation, where you can switch it
# on and off and see the effect on the training. 
class MultiHeadAttentionModel(PosEncLayersAttentionModel):
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        
        # Now it's + 3
        self.input_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 3, self.hidden)))

        self.heads = num_heads
        self.enc_mult_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.num_enc_layers)
        ])
        # We add individual attention norms for each layer now
        self.enc_layer_att_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.num_enc_layers)])        

        self.output_embedding = None # we will use a parameter instead
        self.output_embedding_param = nn.Parameter(torch.zeros((1, 3, self.hidden)))

        # We implement only N=1 layers of the decoder
        self.enc_dec_mult_att = MultiHeadAttention(self.heads, self.hidden)
        self.enc_dec_att_norm = nn.LayerNorm(self.hidden)
        self.dec_feed_forward = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 2), # (11, 128)
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden) # (11, 64)
        )
        self.dec_layer_norm = nn.LayerNorm(self.hidden)

    def forward(self, x): # (13, 3)
        # Left side
        input_embedding = self.input_embedding(x) # (13, 64)

        if self.pos_enc:
            encoding = input_embedding + self.input_pos_enc

        for i in range(self.num_enc_layers):
            # We now use enc_mult_att instead of scaled_dot_product_attention
            multi_headed_attention, _ = self.enc_mult_att[i](
                encoding, 
                encoding, 
                encoding
            )            
            add_and_norm = self.enc_layer_att_norm[i](
                multi_headed_attention + encoding)

            feed_forward = self.enc_feed_forwad[i](add_and_norm)
            add_and_norm = self.enc_layer_norm[i](feed_forward + add_and_norm)

            encoding = add_and_norm

        # Rigth side
        output_embedding = self.output_embedding_param.repeat(x.shape[0], 1, 1)
        
        # We implement only N=1 layers of the decoder
        # We now use enc_mult_att instead of scaled_dot_product_attention
        multi_headed_attention, attention_weights = self.enc_dec_mult_att(
            output_embedding,
            encoding,
            encoding
        ) 
        add_and_norm = self.enc_dec_att_norm(multi_headed_attention + output_embedding)
        
        feed_forward = self.dec_feed_forward(add_and_norm)
        add_and_norm = self.dec_layer_norm(feed_forward + add_and_norm)
        
        linear = self.linear(add_and_norm)

        return linear, attention_weights, self.input_pos_enc
    
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

    def forward(self, query, key, value, mask=False): # encoding: (13, 64), decoding: query=(3, 64)           
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
                mask=mask
            ) # page 5 of Vaswani et al. (2017)

            heads[i] = weighted_sum # (13, 16)
            attention_weights[i] = attention_weight # for visualization
        
        concat = torch.cat(heads, dim=-1) # (13, 64)
        linear = self.W_output(concat) # (13, 64)

        return linear, attention_weights
    
class FullTransformerModel(MultiHeadAttentionModel):
    def __init__(self, en_vocab_size, de_vocab_size, 
                 num_dec_layers=6, max_len=20, **kwargs):
        super().__init__(**kwargs)

        self.max_len = max_len

        # Encoder params
        self.de_vocab_size = de_vocab_size  

        self.input_embedding = nn.Linear(self.de_vocab_size, self.hidden)
        self.input_pos_enc = nn.Parameter(torch.zeros((1, self.max_len, self.hidden)))                

        # Decoder params
        self.en_vocab_size = en_vocab_size
        # Now it's a linear layer
        self.output_embedding = nn.Linear(self.en_vocab_size, self.hidden)        
        self.output_embedding_param = None
        self.output_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.hidden)))

        # Implement a full, multi-layered decoder block
        self.num_dec_layers = num_dec_layers

        # Introduce the first part of the decoder block
        self.dec_mult_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.num_dec_layers)])
        self.dec_layer_att_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.num_dec_layers)])         

        # Change to introduce multiple layers 
        self.enc_dec_mult_att = nn.ModuleList([
            MultiHeadAttention(self.heads, self.hidden)
            for i in range(self.num_dec_layers)])
        self.enc_dec_att_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.num_dec_layers)])        
        self.dec_feed_forwad = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden, self.hidden * 2), # (11, 128)
                nn.ReLU(),
                nn.Linear(self.hidden * 2, self.hidden) # (11, 64)
            )
            for i in range(self.num_dec_layers)
        ])  
        self.dec_layer_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden)
            for i in range(self.num_dec_layers)])      
        
        # We now need to specify that we want en_vocab_size
        self.linear = nn.Linear(self.hidden, self.en_vocab_size)

    # Predict the next token, given the source sentence `de` and what we have
    # predicted so far `en`.
    def forward(self, de, en):
        # Encoder - Left side
        input_embedding = self.input_embedding(de)  
        encoding = input_embedding + self.input_pos_enc

        for i in range(self.num_enc_layers):
            multi_headed_attention, _ = self.enc_mult_att[i](
                encoding, 
                encoding, 
                encoding
            )            
            add_and_norm = self.enc_layer_att_norm[i](
                multi_headed_attention + encoding)

            feed_forward = self.enc_feed_forwad[i](add_and_norm)
            add_and_norm = self.enc_layer_norm[i](feed_forward + add_and_norm)

            encoding = add_and_norm
        
        # Decoder - Right side
        dec_input_emb = self.output_embedding(en)
        decoding = dec_input_emb + self.output_pos_enc

        # New compared to to the previous task
        for i in range(self.num_dec_layers):
            # Decoder Self-Attention
            multi_headed_attention, _ = self.dec_mult_att[i](
                decoding, 
                decoding, 
                decoding, 
                mask=True
            )
            add_and_norm = self.dec_layer_att_norm[i](
                multi_headed_attention + decoding)
            
            
            multi_headed_attention, attention_weights = self.enc_dec_mult_att[i](
                add_and_norm,
                encoding,
                encoding
            ) 
            add_and_norm = self.enc_dec_att_norm[i](multi_headed_attention + add_and_norm)            
            
            feed_forward = self.dec_feed_forwad[i](add_and_norm)
            add_and_norm = self.dec_layer_norm[i](feed_forward + add_and_norm)

        linear = self.linear(add_and_norm)

        return linear, attention_weights      