import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import zipfile

from config import (
    MAX_DECK_SIZE,
    CHARACTERS,
    KEEP_FIELDS
)

def get_data():
    merged = []
    with zipfile.ZipFile("Monthly_2020_11.zip") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json.gz")]
        for name in tqdm(names, desc="Loading runs"):
            with zf.open(name) as zipped, gzip.open(zipped) as f:
                data = json.load(f)
                runs = data if isinstance(data, list) else [data]
                for run in runs:
                    event = run["event"]
                    merged.append({k: event[k] for k in KEEP_FIELDS if k in event})

    return merged

def clean_data(runs):
    new_runs = []
    for run in runs:
        if run['is_ascension_mode'] == False and run['is_daily'] == False:
            new_runs.append(run)

    return new_runs

def get_deck_sizes(decks):
    deck_sizes = [len(deck) for deck in decks]
    average = sum(deck_sizes) / len(decks)
    print(f"Average deck size: {average}")
    print(f"Maximum deck size: {max(deck_sizes)}")
    print(f"Minimum deck size: {min(deck_sizes)}")
    print(f"95th percentille deck size: {np.percentile(deck_sizes, 95)}")


def build_cards(decks):
    cards_with_ids = {"[MASK]": 0, "[EMPTY]": 1}

    # Extract cards, remove upgrades
    all_cards = set([card.strip("+1") for deck in decks for card in deck])
    id = 3
    for card in all_cards:
        cards_with_ids[card] = id
        id += 1
    
    with open("card_ids.json", "w") as f:
        json.dump(cards_with_ids, f)
    
    print(f"Found {len(cards_with_ids)} unique cards")

def encode_deck(deck, card_ids) -> list[int]:
    """
    Encode a deck as a fixed-length integer sequence.
    - Unknown cards map to [MASK] (treated as noise at training time)
    - Decks shorter than max_size are padded with [EMPTY]
    """
    unknown= card_ids["[MASK]"]
    empty = card_ids["[EMPTY]"]
    encoded = [card_ids.get(card.strip("+1"), unknown) for card in sorted(deck)]
    encoded = encoded[:MAX_DECK_SIZE] # truncate
    encoded += [empty] * (MAX_DECK_SIZE - len(encoded)) # pad
    return encoded

def init():
    runs = get_data()
    print(f"Looking at {len(runs)} total runs")
    runs = clean_data(runs)
    print(f"Looking at {len(runs)} A0 runs")
    with open("cleaned_runs.json", "w") as f:
        json.dump(runs, f)
    print("Saved to cleaned_runs.json")
    decks = [run['master_deck'] for run in runs]
    get_deck_sizes(decks)

    wins = [run for run in runs if run['victory']]
    print(f"Analyzing {len(wins)} wins, {len(runs) - len(wins)} losses")

    build_cards(decks)

# Class for use with PyTorch
class SpireDataset(Dataset):
    """
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, card_ids, runs):
        self.card_ids = card_ids
        self.max_size = MAX_DECK_SIZE

        self.samples = []
        for run in runs:
            deck = run["master_deck"]
            if not deck:
                continue

            won = int(run.get("victory", False))
            self.samples.append({
                "deck": encode_deck(deck, card_ids),
                "character": CHARACTERS[run.get("character_chosen", "").upper()],
                "won": won,
            })

        print(f"Dataset: {len(self.samples):,} runs after filtering")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "deck": torch.tensor(s["deck"], dtype=torch.long),
            "character": torch.tensor(s["character"], dtype=torch.long),
            "won": torch.tensor(s["won"], dtype=torch.long),
        }
