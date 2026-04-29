import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    def forward(self, x):
        emb = self.embed(x)
        _, (h, c) = self.lstm(emb)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        emb = self.embed(x)
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        out = out.squeeze(1)
        return self.fc(out), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.enc = Encoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.dec = Decoder(vocab_size, embed_dim, hidden_dim, num_layers)
    def forward(self, src, tgt, teacher_ratio=0.5):
        batch, tgt_len = src.size(0), tgt.size(1)
        hidden, cell = self.enc(src)
        dec_input = tgt[:, 0]
        outputs = torch.zeros(batch, tgt_len, self.dec.fc.out_features)
        for t in range(1, tgt_len):
            out, hidden, cell = self.dec(dec_input, hidden, cell)
            outputs[:, t, :] = out
            teacher = random.random() < teacher_ratio
            dec_input = tgt[:, t] if teacher else out.argmax(1)
        return outputs