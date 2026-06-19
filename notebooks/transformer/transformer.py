import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax
import math
import copy
import warnings

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

## Helper Functions
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

## scaled dot product attention
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Computing the scaled dot-product attention. 

    Input shape:
        query: (batch_size, seq_len, d_k)
        key:   (batch_size, seq_len, d_k)
        value: (batch_size, seq_len, d_v)
    """
    d_k = query.size(-1)
    
    transposed_K = torch.transpose(key, -1, -2)
    scores = torch.matmul(query, transposed_K)
    scaled_scores =  scores / math.sqrt(d_k)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask==0, -1e9)

    attention_weights = softmax(scaled_scores, dim = -1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights

## Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h # d_v = d_k
        
        # linear layers
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Input shape:
            query: tensor of shape [batch_size, seq_len, d_model]
            key:   tensor of shape [batch_size, seq_len, d_model]
            value: tensor of shape [batch_size, seq_len, d_model]
            mask:  tensor of shape [batch_size, 1, seq_len, seq_len]
        """
        nbatches = query.size(0)
        
        # projected tensor, output shape of each: [batch_size, seq_len, d_model]
        q_projected = self.w_q(query)
        k_projected = self.w_k(key)
        v_projected = self.w_v(value)

        # divide model dimension to h heads
        q_heads = q_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        k_heads = k_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        v_heads = v_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        attn_output, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask, self.dropout)

        # combine it back
        concat_output = attn_output.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)
        
        return self.w_out(concat_output)

# Feed-Forward Network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        
        # two linear layers
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input shape:
            x: tensor of shape [batch_size, seq_len, d_model]
        """
        
        return self.w_2(self.dropout(self.w_1(x).relu()))

# Embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # changing the term a bit for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input shape:
            x: Input token embedding tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        
        return self.dropout(x)

# Sublayer Connection
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        """
        Args:
            x: Input tensor from the previous layer [batch_size, seq_len, d_model]
            sublayer_fn: A lambda function or module (like your Attention or FFN)
        """
        return x + self.dropout(sublayer_fn(self.norm(x)))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Attention mask tensor
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, lambda x: self.feed_forward(x))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in the stack sequentially.
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        # We need 3 sublayer connections now!
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: Target sequence tokens [batch_size, tgt_len, d_model]
            memory: The final output vectors from the Encoder [batch_size, src_len, d_model]
            src_mask: Mask to hide padding tokens in the encoder source
            tgt_mask: Causal mask to hide future tokens in the decoder target
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
    
        return self.sublayer[2](x, lambda x: self.feed_forward(x))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: Target sequence tokens
            memory: Final output from the Encoder
            src_mask: Padding mask for the encoder memory
            tgt_mask: Causal look-ahead mask for the decoder target
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            
        return self.norm(x)

# EncoderDecoder - Transformer
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # Token Embeddings + Positional Encoding
        self.tgt_embed = tgt_embed  # Token Embeddings + Positional Encoding
        self.generator = generator  # Final Linear layer + Softmax to project to vocabulary

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

# mini testing
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, h)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

def run_tests():
    for _ in range(10):
        inference_test()

show_example(run_tests)

