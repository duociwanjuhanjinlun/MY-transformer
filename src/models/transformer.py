import torch
import torch.nn as nn
from ..modules.encoder import Encoder
from ..modules.decoder import Decoder
from ..modules.embeddings import TokenEmbedding, PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=5000,
        dropout=0.1
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder and Decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding + Positional Encoding
        src_embedded = self.positional_encoding(self.src_embedding(src))
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        
        # Encoder
        enc_output = self.encoder(src_embedded, src_mask)
        
        # Decoder
        dec_output = self.decoder(tgt_embedded, enc_output, tgt_mask, src_mask)
        
        # Output layer
        output = self.output_layer(dec_output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """生成用于解码器的屏蔽矩阵（防止看到未来信息）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.unsqueeze(0)
    
    @staticmethod
    def create_pad_mask(seq, pad_idx):
        """生成用于处理填充标记的屏蔽矩阵"""
        return (seq == pad_idx).unsqueeze(1).unsqueeze(2)