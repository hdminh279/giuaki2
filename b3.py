
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

corpus = """
Toi dang on thi nhan dang mau. Transformer hoc cach du doan ky tu tiep theo.
Mot mo hinh nho nhung day du thanh phan: token embedding, positional encoding,
multi-head self-attention, feed-forward, residual + layer norm, causal mask va cross-entropy.
"""


chars = sorted(list(set(corpus)))
stoi  = {ch:i for i, ch in enumerate(chars)}
itos  = {i:ch for ch, i in stoi.items()}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

data = encode(corpus)
device = "cuda" if torch.cuda.is_available() else "cpu"


block_size = 64   # context length (sequence length)
batch_size = 64
def get_batch(split="train"):
    # simple random slicing (no real split here; good enough for demo)
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x  = torch.stack([data[i:i+block_size] for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])  # next char targets
    return x.to(device), y.to(device)


class PositionalEncoding(nn.Module):
    # Sinusoidal positional encoding (classic)
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x):  # x: (T, B, d_model)
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=4, d_ff=4*256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout,
            batch_first=False, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # causal mask cache
        self.register_buffer("causal_mask", torch.triu(torch.ones(block_size, block_size)*float('-inf'), diagonal=1))

    def forward(self, idx):
        # idx: (B, T) integers
        B, T = idx.shape
        x = self.tok_emb(idx).transpose(0,1)          # (T, B, d_model)
        x = self.pos_enc(x)                           # add sinusoidal positions
        mask = self.causal_mask[:T, :T]               # (T, T) upper-triangular -inf above diagonal
        x = self.encoder(x, mask)                     # Transformer stack
        x = self.ln_f(x)                              # (T, B, d_model)
        logits = self.head(x).transpose(0,1)          # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        # greedy sampling for demo
        for _ in range(max_new_tokens):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
            logits = self.forward(idx)[:, -1, :]      # last time step
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ----------------------------
# 4) Training
# ----------------------------
vocab_size = len(chars)
model = TransformerLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()  # will flatten logits & targets per token

def train(steps=1000):
    model.train()
    for step in range(1, steps+1):
        x, y = get_batch()
        logits = model(x)                 # (B, T, V)
        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), y.reshape(B*T))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.3f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    train(steps=600)  # ~a minute on CPU; faster on GPU
    model.eval()
    prompt = "Transformer "
    out = model.generate(encode(prompt).unsqueeze(0).to(device), max_new_tokens=120)
    print("\n=== SAMPLE ===")
    print(decode(out[0].cpu()))
