import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# Hyperparameters
# =============================================================================
batch_size    = 48      # number of independent sequences processed in parallel
block_size    = 128     # maximum context length (in characters) for predictions
max_iters     = 5000    # total training steps
eval_interval = 500     # how often to evaluate and print train/val loss
learning_rate = 3e-4    # AdamW learning rate
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 50     # number of batches averaged when estimating loss
n_embd        = 256     # embedding dimension (each token becomes a 384-d vector)
n_head        = 4       # number of attention heads (head_size = n_embd // n_head = 64)
n_layer       = 4       # number of stacked transformer blocks
dropout       = 0.2     # dropout probability (regularisation — set to 0 for inference)
# =============================================================================

torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
# Place yoda_dialogue.txt in the same directory as this script.
# Generate it with the Yoda Dataset Generator artifact if you haven't already.
with open('yoda_dialogue.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset loaded: {len(text):,} characters")

# -----------------------------------------------------------------------------
# Character-level tokeniser
# -----------------------------------------------------------------------------
# Build a vocabulary from every unique character in the dataset.
# This is the simplest possible tokeniser: one integer per character.
chars      = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}   # char  -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int   -> char

encode = lambda s: [stoi[c] for c in s]                # string -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])        # list[int] -> string

# -----------------------------------------------------------------------------
# Train / validation split  (90% train, 10% val)
# -----------------------------------------------------------------------------
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# -----------------------------------------------------------------------------
# Batch sampler
# -----------------------------------------------------------------------------
def get_batch(split: str):
    """
    Return a random batch of (inputs, targets) tensors.

    For each sequence in the batch we pick a random starting position,
    then slide a one-step offset to create the targets so that position t
    of x predicts position t of y (i.e. the next character).
    """
    source = train_data if split == 'train' else val_data
    # Random starting indices — one per sequence in the batch
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x  = torch.stack([source[i:i + block_size]     for i in ix])
    y  = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Loss estimator  (no gradient tracking needed here)
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss() -> dict:
    """
    Average the cross-entropy loss over `eval_iters` batches for both splits.
    Switches the model to eval mode (disables dropout) during evaluation.
    """
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =============================================================================
# Model architecture
# =============================================================================

class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, head_size: int):
        super().__init__()
        # Linear projections for keys, queries, and values — no bias (common GPT convention)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask registered as a buffer (not a parameter — won't be updated)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                              # batch, time-steps, channels

        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # Scaled dot-product attention scores
        # Scaling by 1/sqrt(head_size) keeps the variance of the dot product stable
        scale = k.shape[-1] ** -0.5
        wei   = q @ k.transpose(-2, -1) * scale        # (B, T, T)

        # Causal mask: tokens can only attend to earlier positions (and themselves)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)                    # (B, T, T)  — rows sum to 1
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v   = self.value(x)                             # (B, T, hs)
        out = wei @ v                                   # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """
    Several attention heads running in parallel, then concatenated and projected.
    Running multiple heads lets the model attend to different parts of the context
    for different reasons simultaneously.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Project the concatenated head outputs back to n_embd
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate all head outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network applied after attention.
    Expands to 4× the embedding dimension internally (as in the original
    'Attention Is All You Need' paper), then contracts back.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    One transformer block: self-attention followed by a feed-forward network.
    Both sub-layers use residual connections and pre-layer normalisation
    (Pre-LN is slightly more stable than the original Post-LN formulation).
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)   # communication
        self.ffwd = FeedForward(n_embd)                     # computation
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections: x + f(LayerNorm(x))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT trained at the character level.
    Architecture: token embedding + positional embedding -> N transformer
    blocks -> final LayerNorm -> linear head over vocabulary.
    """

    def __init__(self):
        super().__init__()
        # Token embedding: maps each character index to a dense vector
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Positional embedding: learned position encodings for positions 0..block_size-1
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # Final layer norm before the output projection
        self.ln_f   = nn.LayerNorm(n_embd)
        # Linear head: maps from embedding space to logits over the vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialise weights: small normal for linear/embedding layers, zeros for biases
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                            # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x       = tok_emb + pos_emb                                          # (B, T, C)
        x       = self.blocks(x)                                             # (B, T, C)
        x       = self.ln_f(x)                                               # (B, T, C)
        logits  = self.lm_head(x)                                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time dimensions for cross-entropy
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively sample `max_new_tokens` characters given a context `idx`.
        At each step we crop the context to the last `block_size` tokens so
        positional embeddings stay in range, then sample the next character
        from the softmax distribution over the vocabulary.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                 # crop to context window
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]                    # last time-step: (B, C)
            probs     = F.softmax(logits, dim=-1)           # (B, vocab_size)
            idx_next  = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx       = torch.cat((idx, idx_next), dim=1)   # (B, T+1)
        return idx


# =============================================================================
# Training
# =============================================================================

model = GPTLanguageModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params / 1e6:.2f}M  |  device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):

    # Periodically evaluate and report train/val loss
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step:>5}: train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)   # set_to_none=True is slightly faster than zero_grad()
    loss.backward()
    optimizer.step()

# =============================================================================
# Generation
# =============================================================================

# Seed with a single zero token and generate 500 characters
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("\n--- Generated text ---")
print(generated)

# Uncomment to write a longer sample to a file:
with open('yoda_generated.txt', 'w') as f:
    f.write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))