import json
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess import SpireDataset
from diffusion import MaskDiffusion
from denoise import build_model

from config import (
    BATCH_SIZE,
    LR,
    EPOCHS,
    WIN_CONDITION,
    WEIGHTED_SAMPLING,
    T
)

def train():
    # Device set up
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
 
    # Data set up
    runs = json.load(open("cleaned_runs.json"))
    card_ids = json.load(open("card_ids.json"))
    mask_id = card_ids["[MASK]"]
 
    dataset = SpireDataset(card_ids, runs)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
 
    if WEIGHTED_SAMPLING:
        won_labels = [dataset.samples[i]["won"] for i in train_set.indices]
        n_wins = sum(won_labels)
        n_losses = len(won_labels) - n_wins
        weights = [1.0 / n_wins if w else 1.0 / n_losses for w in won_labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE)
 
    # Model and diffusion
    model = build_model(len(card_ids)).to(device)
    diffusion = MaskDiffusion(T=T, mask_id=mask_id, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
 
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x0   = batch["deck"].to(device)
            char = batch["character"].to(device)
            won  = batch["won"].to(device)
 
            if WIN_CONDITION:
                # 10% of the time, mask win label so model learns unconditional too
                drop = torch.rand(won.size(0), device=device) < 0.1
                won_in = won.clone()
                won_in[drop] = 0   # treat as "unknown" — simple approach
 
            t     = diffusion.sample_timesteps(x0.size(0))
            x_t   = diffusion.q_sample(x0, t)
 
            logits = model(x_t, t, char, won_in if WIN_CONDITION else won)
            # Cross-entropy loss only at masked positions
            mask   = (x_t == mask_id)                             # (B, L)
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(
                logits[mask],    # (N_masked, V)
                x0[mask],        # (N_masked,)
            )
 
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
 
        avg_train = total_loss / len(train_loader)
 
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x0   = batch["deck"].to(device)
                char = batch["character"].to(device)
                won  = batch["won"].to(device)
                t    = diffusion.sample_timesteps(x0.size(0))
                x_t  = diffusion.q_sample(x0, t)
                logits = model(x_t, t, char, won)
                mask   = (x_t == mask_id)
                if mask.sum() == 0:
                    continue
                val_loss += F.cross_entropy(logits[mask], x0[mask]).item()
        avg_val = val_loss / len(val_loader)
 
        print(f"Epoch {epoch:3d} | train loss: {avg_train:.4f} | val loss: {avg_val:.4f}")
        with open("loss.txt", "a") as f:
            f.write(f"Epoch {epoch:3d} | train loss: {avg_train:.4f} | val loss: {avg_val:.4f}\n")
 
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "card_ids": card_ids,
        }, "checkpoint.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"model": model.state_dict(), "card_ids": card_ids},
                       "best_model.pt")
            print("Saved best model")
 
    return model, card_ids, diffusion
 