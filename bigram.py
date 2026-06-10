import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# Hyperparameters
# =============================================================================
batch_size    = 32      # number of independent sequences processed in parallel
block_size    = 8       # maximum context length (bigram only looks back 1 step,
                        # but block_size defines the sequence length for batching)
max_iters     = 3000    # total training steps
eval_interval = 300     # how often to print train/val loss
learning_rate = 1e-2    # higher LR is fine for this simple model
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200     # batches averaged when estimating loss
# =============================================================================

torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
with open('yoda_dialogue.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset loaded: {len(text):,} characters")

# -----------------------------------------------------------------------------
# Character-level tokeniser
# -----------------------------------------------------------------------------
# Vocabulary = every unique character in the dataset.
# Each character maps to a unique integer index.
chars      = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}   # char  -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int   -> char

encode = lambda s: [stoi[c] for c in s]         # string   -> list[int]
decode = lambda l: ''.join([itos[i] for i in l]) # list[int] -> string

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
    Return a random batch of (inputs x, targets y) tensors.
    For each sequence, targets are x shifted one position to the right,
    so position t of x predicts position t of y (the next character).
    """
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x  = torch.stack([source[i:i + block_size]         for i in ix])
    y  = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Loss estimator
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss() -> dict:
    """
    Average cross-entropy loss over `eval_iters` batches for both splits.
    Dropout (if any) is disabled during evaluation via model.eval().
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
# Model
# =============================================================================

class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model: a bigram model.

    A single embedding table of shape (vocab_size, vocab_size) is used.
    Each input token index directly looks up a row, which is treated as
    the logit distribution over the *next* token — no attention, no
    hidden state, just a learned table of token-to-token transition scores.

    This is a useful baseline before adding the transformer blocks from gpt.py.
    Expected val loss after training: ~2.5  (compare to ~1.5 for the full GPT).
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding doubles as the prediction head: row i gives logits for
        # "what character comes after character i?"
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        logits = self.token_embedding_table(idx)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time for cross-entropy: expects (N, C) and (N,)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively sample `max_new_tokens` characters.
        The bigram model only uses the last token for prediction, so the
        entire history is carried along purely for output purposes.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits    = logits[:, -1, :]                        # last time-step: (B, vocab_size)
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx       = torch.cat((idx, idx_next), dim=1)       # (B, T+1)
        return idx

# =============================================================================
# Training
# =============================================================================

model = BigramLanguageModel(vocab_size).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M  |  device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:>5}: train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =============================================================================
# Generation
# =============================================================================

context  = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("\n--- Generated text ---")
print(generated)