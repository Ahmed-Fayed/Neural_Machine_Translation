import torch
import torch.nn as nn
import math



class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        d_model: embedding vector size (dimension)
        vocab_size: vocabulary size
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x) -> nn.Embedding:
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        d_model: embedding vector size (dimension)
        seq_len: Max sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # @todo: Why drop out here??

        # Create a matrix for positional encoding with shape (seq_len, d_mdoel)
        pos_encode = torch.zeros(seq_len, d_model)

        # create vector of sequence positions with shape (seq_len, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # create division term vector with shape (half_d_model,)
        div = torch.exp(torch.arange(0, d_model, 2).float * (-math.log(10000.0) / d_model))

        # apply sin to even positions: sin(position * (10000 ** (2i / d_model))
        pos_encode[:, 0::2] = torch.sin(positions * div)
        # apply cos to odd positions:  cos(position * (10000 ** (2i / d_model))
        pos_encode[:, 1::2] = torch.cos(positions * div)

        pos_encode = pos_encode.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pos_encode', pos_encode)
    

    def forward(self, x):
        """
        selecting a subset of the positional encoding matrix. The selected subset includes all batches,
        the positional encoding values up to the length of the input sequence, and all dimensions of the positional encoding values.
        """
        x = x + (self.pos_encode[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float=1**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def fowward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True) # Multiplied
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True) # Added
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and4 B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        self.dropout = nn.Dropout(dropout)

        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Write a avery low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # (batch, h, seq_len, seq_len) # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
       # return attention scores which can be used for visualization
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, h * d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (batch, seq_len, h * d_k) --> (batch, seq_len, h * d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # Original Paper --> x + self.dropout(self.norm(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention_block: MultiheadAttentionBlock, fead_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.multihead_attention_block = multihead_attention_block
        self.fead_forward_block = fead_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
    
    def forward(self, x, mask):
       x = self.residual_connection[0](x, lambda x: self.multihead_attention_block(x, x, x, mask))
       x = self.residual_connection[1](x, self.fead_forward_block)
       return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.multihead_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.drop
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.multihead_attention_block(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.project(x), dim=-1)



class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, target_embed: InputEmbedding, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder: decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        # (batch, seq_len, d_model)
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, x, encoder_output, src_mask, target_mask):
        # (batch, seq_len, d_model)
        x = self.target_embed(x)
        x = self.target_pos(x)
        return self.decoder(x, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        # (Batch, seq_len, vocab_size)
        return self.projection_layer(x)



def build_transformers(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformers:
    # Create the embeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)

    # Create the positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Create the transformers architecture
    transformers = Transformers(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)


    # Initialize the parameters
    for param in transformers.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    return Transformers


