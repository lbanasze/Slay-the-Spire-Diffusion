import random
import torch

from diffusion import MaskDiffusion
from training import build_model

from config import (
    CHARACTERS,
    BATCH_SIZE,
    LR,
    EPOCHS,
    MAX_DECK_SIZE,
    N_SAMPLES,
    WIN_CONDITION,
    T
)

def sample_decks(checkpoint_path, character):
    # Device set up
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    card_ids = ckpt["card_ids"]
    id_to_tok = {v: k for k, v in card_ids.items()}

    mask_id   = card_ids["[MASK]"]
    empty_id  = card_ids["[EMPTY]"]

    model = build_model(len(card_ids)).to(device)
    model.load_state_dict(ckpt["model"])

    diffusion = MaskDiffusion(T=T, mask_id=mask_id, device=device)

    char_idx = CHARACTERS[character]
    n = N_SAMPLES

    character = torch.full((n,), char_idx, dtype=torch.long, device=device)
    won       = torch.full((n,), int(WIN_CONDITION), dtype=torch.long, device=device)

    samples = diffusion.sample(model, n, MAX_DECK_SIZE, character, won)
    print("Raw sample:", samples[0])

    print(f"\nGenerated {n} decks (character={character}, win_cond={WIN_CONDITION}):\n")
    for i, deck_ids in enumerate(samples):
        cards = [id_to_tok[idx.item()] for idx in deck_ids
                 if id_to_tok[idx.item()] not in ("[EMPTY]", "[MASK]", "[PAD]")]
        print(f"Deck {i+1}: {cards}\n")


def complete_deck(checkpoint_path, character, known_cards):
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    ckpt = torch.load(checkpoint_path, map_location=device)
    card_ids = ckpt["card_ids"]
    id_to_tok = {v: k for k, v in card_ids.items()}

    mask_id  = card_ids["[MASK]"]
    empty_id = card_ids["[EMPTY]"]

    model = build_model(len(card_ids)).to(device)
    model.load_state_dict(ckpt["model"])

    diffusion = MaskDiffusion(T=T, mask_id=mask_id, device=device)

    char_idx  = CHARACTERS[character]
    n = 5

    char_tensor = torch.full((n,), char_idx, dtype=torch.long, device=device)
    won_tensor  = torch.full((n,), int(WIN_CONDITION), dtype=torch.long, device=device)

    # Encode known cards; unknown names fall back to [MASK]
    encoded_known = [card_ids.get(c, mask_id) for c in known_cards]
    n_known = len(encoded_known)

    # Scatter known cards across random positions so masked slots appear throughout the sequence, not just at the end where the model expects [EMPTY]
    positions = sorted(random.sample(range(MAX_DECK_SIZE), n_known))
    padded = [mask_id] * MAX_DECK_SIZE
    for pos, card_id in zip(positions, encoded_known):
        padded[pos] = card_id

    x_init = torch.tensor(padded, dtype=torch.long, device=device).unsqueeze(0).expand(n, -1).contiguous()
    fixed_mask = torch.zeros(n, MAX_DECK_SIZE, dtype=torch.bool, device=device)
    fixed_mask[:, positions] = True

    samples = diffusion.sample(model, n, MAX_DECK_SIZE, char_tensor, won_tensor, x_init=x_init, fixed_mask=fixed_mask)

    print(f"\nCompleted {n} decks (fixed {n_known} cards, generated {MAX_DECK_SIZE - n_known}):\n")
    all_decks = []
    for i, deck_ids in enumerate(samples):
        cards = [id_to_tok[idx.item()] for idx in deck_ids
                 if id_to_tok[idx.item()] not in ("[EMPTY]", "[MASK]", "[PAD]")]
        all_decks.append(cards)

    return all_decks
