import torch
import torch.nn as nn
import torch.nn.functional as F
import random

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# ---------- DATA ----------
# data = [
#     "Pakistan is a beautiful country with a rich history and culture.",
#     "The people of Pakistan are very kind and hospitable.",
#     "Pakistan has stunning mountains, rivers, and valleys.",
#     "The country has a strong and brave army.",
#     "Pakistani youth are talented and hardworking.",
#     "Pakistan has unique traditions and cultural diversity.",
#     "The land of Pakistan is fertile and good for agriculture.",
#     "Many languages and cultures live together in harmony in Pakistan.",
#     "Pakistan has shown courage and strength in difficult times.",
#     "We are proud of Pakistan and hope for its bright future. 🇵🇰"
# ]

# add <End> token to each sentence
# data = [s + "<End>" for s in data]
# text = " ".join(data)
# print("Full text:\n", text)

# # build vocabulary
# words = list(set(text.split()))
# wordsize = len(words)
# print("Vocabulary size:", wordsize)

# wordidx = {w: i for i, w in enumerate(words)}
# idx2word = {i: w for w, i in wordidx.items()}

# # encode text as indices
# data_tensor = torch.tensor([wordidx[w] for w in text.split()], dtype=torch.long)
# print("Total data tokens:", len(data_tensor))

# ---------- HYPERPARAMETERS ----------
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1000  # reduced for quick testing

# ---------- BATCH FUNCTION ----------
def get_batch(batch_size=16):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x, y

# ---------- MODEL COMPONENTS ----------
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2, -1) / (C ** 0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * embedding_dim, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ff = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ---------- TINY GPT MODEL ----------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(wordsize, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, wordsize)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


start_word = "country"
context = torch.tensor([[wordidx[start_word]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))