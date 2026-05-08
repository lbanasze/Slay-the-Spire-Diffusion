import math
import torch
import torch.nn as nn

from config import (
    CHARACTERS,
    MAX_DECK_SIZE,
    N_LAYERS,
    N_HEADS,
    DROPOUT,
    D_MODEL
)

def sinusoidal_emb(t, d):
    """
    Sinusoidal timestep embedding. Encodes the scalar timestep t into a
    d-dimensional vector using sine and cosine at different frequencies.
 
    This gives the model a smooth, structured sense of "how noisy is the
    input right now?" without just feeding it a raw integer.
 
    Shape: (B,) -> (B, d)
    """
    assert d % 2 == 0, "d_model must be even for sinusoidal embedding"
    half = d // 2
    # Frequencies decrease geometrically: high-freq captures fine timestep
    # differences, low-freq captures coarse position in the schedule
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )                                                      # (d/2,)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)    # (B, d/2)
    return torch.cat([args.sin(), args.cos()], dim=-1)     # (B, d)
 
 
class ConditioningProjector(nn.Module):
    """
    Projects all conditioning signals (timestep, character, win label) into
    a single d-dimensional vector that gets injected into each transformer layer.
 
    Using a dedicated projector with its own LayerNorm is better than simple
    addition because:
      - Each conditioning signal starts at a different scale
      - LayerNorm stabilizes training early on when embeddings are random
      - A learned linear projection lets the model decide how much each
        conditioning signal should influence each layer
    """
 
    def __init__(self, d: int, n_chars: int):
        super().__init__()
        # Timestep: sinusoidal -> linear projection
        self.time_proj = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),               # SiLU (swish) works better than ReLU for
            nn.Linear(d * 2, d),     # conditioning signals empirically
        )
        # Character: learned embedding (4 characters)
        self.char_emb = nn.Embedding(n_chars, d)
        # Win/loss: learned embedding (2 classes)
        self.win_emb  = nn.Embedding(2, d)
        # Normalize the combined conditioning vector
        self.norm = nn.LayerNorm(d)
 
    def forward(
        self,
        t: torch.Tensor,        # (B,)
        character: torch.Tensor, # (B,)
        won: torch.Tensor,       # (B,)
        d: int,
    ) -> torch.Tensor:
        t_emb   = self.time_proj(sinusoidal_emb(t, d))  # (B, d)
        ch_emb  = self.char_emb(character)               # (B, d)
        win_emb = self.win_emb(won)                      # (B, d)
        # Sum and normalize — all three signals contribute equally before
        # the transformer layers decide how to use them
        return self.norm(t_emb + ch_emb + win_emb)       # (B, d)
 
 
class TransformerBlock(nn.Module):
    """
    A single transformer encoder layer with conditioning injection.
 
    Standard TransformerEncoderLayer adds conditioning by simple addition
    before the block. We do it properly: condition is added after the
    first LayerNorm (pre-norm architecture), which is more stable to train.
 
    Pre-norm (used here):   x -> LayerNorm -> Attention -> + residual
    Post-norm (original):   x -> Attention -> + residual -> LayerNorm
 
    Pre-norm is now standard in modern transformers (GPT, BERT variants)
    because gradients flow more cleanly through the residual stream.
    """
 
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),           # GELU is standard in modern transformers
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )
        # Learned scale for conditioning injection — starts near zero so
        # early training is stable (conditioning is gradually turned on)
        self.cond_scale = nn.Linear(d, d, bias=False)
 
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond is (B, d) — unsqueeze to (B, 1, d) to broadcast across positions
        cond = self.cond_scale(cond).unsqueeze(1)   # (B, 1, d)
 
        # Self-attention with pre-norm and conditioning
        h = self.norm1(x + cond)                    # inject cond before attention
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out                            # residual connection
 
        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))              # residual connection
        return x                                    # (B, L, d)
 
 
class DenoisingTransformer(nn.Module):
    """
    Predicts the original (unmasked) token at every position given:
      - x_t       : (B, L)  the partially masked deck at timestep t
      - t         : (B,)    the current diffusion timestep
      - character : (B,)    which character (Ironclad/Silent/Defect/Watcher)
      - won       : (B,)    win/loss label for classifier-free guidance
 
    Returns logits : (B, L, V) — a distribution over vocab at every slot.
    The model only needs to be correct at masked positions during training,
    but predicting all positions lets us use the output cleanly at inference.
    """
 
    def __init__(self, card_ids_size, n_chars):
        super().__init__()
        d = D_MODEL
        L = MAX_DECK_SIZE
 
        # --- Input embeddings ---
        # Token embedding: maps each card id to a d-dimensional vector.
        # [MASK], [EMPTY], and real cards all get their own learned vectors.
        self.token_emb = nn.Embedding(card_ids_size, d)
 
        # Positional embedding: even though decks are unordered, the transformer
        # needs some way to distinguish slot 1 from slot 2. We use learned
        # positional embeddings. The model will learn to ignore order and
        # attend based on content, but needs positions to do cross-slot attention.
        self.pos_emb = nn.Embedding(L, d)
 
        # Input LayerNorm: stabilizes the sum of token + positional embeddings
        self.input_norm = nn.LayerNorm(d)
 
        # --- Conditioning ---
        self.conditioner = ConditioningProjector(d, n_chars)
 
        # --- Transformer layers ---
        # Each layer gets the conditioning signal injected independently,
        # allowing different layers to use different aspects of the condition
        self.layers = nn.ModuleList([
            TransformerBlock(d, N_HEADS, DROPOUT)
            for _ in range(N_LAYERS)
        ])
 
        # Final LayerNorm before output projection (standard in pre-norm transformers)
        self.out_norm = nn.LayerNorm(d)
 
        # --- Output projection ---
        # Maps each position's hidden state to a distribution over vocab
        self.out = nn.Linear(d, card_ids_size)
 
        # Initialize output projection to near-zero so early predictions
        # are close to uniform — this stabilizes the first few training steps
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, std=0.02)
 
    def forward(
        self,
        x_t: torch.Tensor,       # (B, L)
        t: torch.Tensor,          # (B,)
        character: torch.Tensor,  # (B,)
        won: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        B, L = x_t.shape
        d = self.token_emb.embedding_dim
 
        # --- Build input representation ---
        pos = torch.arange(L, device=x_t.device).unsqueeze(0)  # (1, L)
        h = self.input_norm(
            self.token_emb(x_t) + self.pos_emb(pos)            # (B, L, d)
        )
 
        # --- Compute conditioning vector ---
        # Single vector per sample that summarizes timestep + character + outcome
        cond = self.conditioner(t, character, won, d)           # (B, d)
 
        # --- Pass through transformer layers ---
        # Each layer attends across all card slots and uses the conditioning
        # signal to modulate its behavior
        for layer in self.layers:
            h = layer(h, cond)                                  # (B, L, d)
 
        # --- Project to vocab ---
        return self.out(self.out_norm(h))                       # (B, L, V)
 
 
def build_model(card_ids_size) -> nn.Module:
    n_chars = len(CHARACTERS)
    return DenoisingTransformer(card_ids_size, n_chars)
 