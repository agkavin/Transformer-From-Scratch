import torch
import torch.nn as nn
import math

# to create emmbeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocab size
        self.embedding = nn.Embedding  (vocab_size, d_model) # embedding layer

#  convert raw token index to embedding
    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.d_model)
    
# to create positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int,seq_len:int,dropout:float=0.1):
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.seq_len = seq_len # sequence length
        self.dropout = nn.Dropout(dropout) # to avoid overfitting
        # matrix to store positional encoding
        pe = torch.zeros(seq_len, d_model)

        # compute positional encoding
        
        #formula for positional encoding is if even position use sin(postion * 1/10000^(2i/d_model)) else use cos(postion * 1/10000^(2i/d_model))

        #vector of length seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # 1/10000^(2i/d_model) instead log is used for simplification  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # register buffer to store positional encoding 
        self.register_buffer('pe', pe)

    # add positional encoding to input Embedding
    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].requires_grad_(False) 
        # requires_grad_(False) to avoid gradient calculation as the positional encoding is fixed by formula here

        # dropout to avoid overfitting
        return self.dropout(x)
    
# Layer Normalization is used to normalize the inputs to a layer
class LayerNormalization(nn.Module):
    # eps to avoid division by zero
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # aplha(multiplicative factor) and bais(additive factor) are learnable parameters
        self.aplha = nn.Parameter(torch.ones(1))
        self.bais = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.aplha * (x - mean) / (std + self.eps) + self.bais
    
# feed forward block is used to forward the input  
class FeedForwardBlock(nn.Module):
    # d_ff is the dimension of the hidden layer
    # formula for FFN is max(0, xW1+b1)W2+b2
    # W1 and b1 are learnable parameters
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        # 1 hidden layer and 1 output layer

        # W1 and b1
        self.linear1 = nn.Linear(d_model, d_ff)
        # W2 and b2
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Len, D_model) -> (Batch, Len, D_ff) -> (Batch, Len, D_model)
        x = self.dropout(torch.relu(self.linear1(x)))
        return self.linear2(x)
    
# multi head attention is used to implement attention
class MultiHeadAttention(nn.Module):
    # h is the number of heads
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        # d_k is the dimension of the key and value
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        # WQ, WK, WV are learnable parameters
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        # the formula is query * key^T / sqrt(d_k)
        d_k = query.shape[-1]
        # (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores= attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask=None):
        # mask is used to avoid attention on padding tokens
        # (Batch, Len, D_model) -> (Batch, Len, D_model) -> (Batch, Len, D_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

     # (Batch, Len, D_model) -> (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k) -> (Batch, Seq_Len, D_model)
        # contiguous is used to make the tensor contiguous in memory
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        return self.w_o(x)
    
# residual connection is used to create connection between layers
class ResidualConnection(nn.Module):
    def __init__(self,  dropout:float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        return x + self.dropout(sublayer(self.norm(x)))
    
# encoder block is used to implement encoder layers
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        #src_mask is used to avoid attention on padding tokens
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# Encoder class is used to implement encoder
class Encoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

# decoder block is used to implement decoder layers
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(([ResidualConnection(dropout) for _ in range(3)]))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, D_model)
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

# decoder class is used to implement decoder
class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

# ProjectionLayer is used to map back to vocab
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, vocab_size)
        # log_softmax is used for better numerical stability
        return torch.log_softmax(self.projection(x), dim=-1)
    

# Transformer class is used to implement transformer

class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder:Decoder, src_embedding:InputEmbeddings, tgt_embedding:InputEmbeddings,       src_pos:PositionalEncoding,
                 tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (Batch, Seq_Len) -> (Batch, Seq_Len, D_model)
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (Batch, Seq_Len) -> (Batch, Seq_Len, D_model)
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (Batch, Seq_Len, D_model) -> (Batch, Seq_Len, vocab_size)
        return self.projection_layer(x)
    


def build_transformer(src_vocab_size:int, tgt_vocab_size:int,src_seq_len:int, tgt_seq_len:int ,d_model:int = 512, N: int = 6, h: int = 8, dropout:float= 0.1, dim_ff:int = 2048):
    # here N is the number of encoder and decoder blocks
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    # positional encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model,dim_ff,dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model,dim_ff,dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    
    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    #create transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    # Initialize parameters
    # this ensures that the model starts with well-distributed weights, which are better than purely random ones for faster and more stable learning.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

test_transformer = build_transformer(10,10,10,10)
print(test_transformer)
