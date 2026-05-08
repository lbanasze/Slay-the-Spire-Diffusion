from collections import Counter
import json

from training import train
from sample import sample_decks, complete_deck
from preprocess import SpireDataset

# train()

def test_deck_generation(character, start_deck, model_path, output_path):
    end_decks = complete_deck(
        model_path,
        character,
        start_deck
    )

    output = []
    deck_number = 1
    for end_deck in end_decks:
        c1 = Counter(start_deck)
        c2 = Counter(end_deck)

        diff = c2-c1
        output.append(f"Deck {deck_number} added {len(list(diff.elements()))} cards: {list(diff.elements())}")
        deck_number += 1
    
    json.dump(output, open(output_path, "w"))

def main():
    test_deck_generation("IRONCLAD", json.load(open("trials/ironclad/starting_deck.json")), "unweighted_models/45.pt", "trials/ironclad/unweighted_generated_decks.json")
    test_deck_generation("IRONCLAD", json.load(open("trials/ironclad/starting_deck.json")), "weighted_models/38.pt", "trials/ironclad/weighted_generated_decks.json")

    test_deck_generation("THE_SILENT", json.load(open("trials/silent/starting_deck.json")), "unweighted_models/45.pt", "trials/silent/unweighted_generated_decks.json")
    test_deck_generation("THE_SILENT", json.load(open("trials/silent/starting_deck.json")), "weighted_models/38.pt", "trials/silent/weighted_generated_decks.json")

    test_deck_generation("DEFECT", json.load(open("trials/defect/starting_deck.json")), "unweighted_models/45.pt", "trials/defect/unweighted_generated_decks.json")
    test_deck_generation("DEFECT", json.load(open("trials/defect/starting_deck.json")), "weighted_models/38.pt", "trials/defect/weighted_generated_decks.json")

    test_deck_generation("WATCHER", json.load(open("trials/watcher/starting_deck.json")), "unweighted_models/45.pt", "trials/watcher/unweighted_generated_decks.json")
    test_deck_generation("WATCHER", json.load(open("trials/watcher/starting_deck.json")), "weighted_models/38.pt", "trials/watcher/weighted_generated_decks.json")

main()