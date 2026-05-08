"""
Reference:
Structured Denoising Diffusion Models in Discrete State-Spaces
https://arxiv.org/pdf/2107.03006
https://github.com/cloneofsimo/d3pm
"""

import torch
from tqdm import tqdm

class MaskDiffusion:
    """
    Forward process: independently mask each card position with probability beta_t = t / T   (linear schedule)

    Reverse process: a neural network predicts x_0 from x_t, then sample x_{t-1} using the posterior.
    """

    def __init__(self, T, mask_id, device):
        self.T = T # number of diffusion timesteps
        self.mask_id = mask_id # card id for masked cards
        self.device = device

        t = torch.arange(0, T + 1, dtype=torch.float32)
        self.alpha_bar = (1.0 - t / T).to(device) # how much of the original signal survives

    def q_sample(self, x0, t) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) by masking tokens independently. 
        Each token either stays the same, or becomes MASK
        x0: (B, L) int tensor of token ids
        t:  (B,)   int tensor of timesteps in [1, T]
        Returns x_t: (B, L)
        """
        alpha = self.alpha_bar[t].unsqueeze(1)         
        keep = torch.bernoulli(alpha.expand_as(x0.float())).bool()
        x_t = x0.clone()
        x_t[~keep] = self.mask_id
        return x_t

    def sample_timesteps(self, batch_size) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (batch_size,), device=self.device)

    def p_sample_step(
        self,
        model,
        x_t,
        t,
        character,
        won,
    ) -> torch.Tensor:
        """
        One reverse step: predict x_0 from x_t, then sample x_{t-1}.
        Uses the closed-form posterior for absorbing diffusion.
        """
        t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        with torch.no_grad():
            logits = model(x_t, t_tensor, character, won)   # (B, L, V)
            x0_pred = logits.argmax(dim=-1)                 # (B, L) hard prediction

        if t == 1:
            return x0_pred

        # Posterior: positions that are masked either stay masked or unmasked
        alpha_t   = self.alpha_bar[t]
        alpha_tm1 = self.alpha_bar[t - 1]

        # Probability that a currently-masked token came from a real token and should be unmasked at t-1
        p_unmask = (alpha_tm1 - alpha_t) / (1.0 - alpha_t + 1e-8)
        p_unmask = p_unmask.clamp(0, 1)

        x_tm1 = x_t.clone()
        masked_positions = (x_t == self.mask_id)
        unmask = masked_positions & (torch.rand_like(x_t.float()) < p_unmask)
        x_tm1[unmask] = x0_pred[unmask]
        return x_tm1

    @torch.no_grad()
    def sample(
        self,
        model,
        n,
        seq_len,
        character,
        won,
        x_init=None,
        fixed_mask=None,
    ) -> torch.Tensor:
        """Start fully masked (or from x_init), denoise to t=0.
        fixed_mask: (n, seq_len) bool tensor — positions to keep fixed throughout."""
        model.eval()
        if x_init is not None:
            x = x_init.clone()
        else:
            x = torch.full((n, seq_len), self.mask_id, dtype=torch.long, device=self.device)
        for t in tqdm(range(self.T, 0, -1), desc="Sampling", leave=False):
            x = self.p_sample_step(model, x, t, character, won)
            if fixed_mask is not None:
                x[fixed_mask] = x_init[fixed_mask]
        return x