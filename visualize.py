from collections import defaultdict
import json
import matplotlib.pyplot as plt
import re

def loss():
    epochs, train_losses, val_losses = [], [], []

    with open("loss.txt") as f:
        for line in f:
            m = re.search(r"Epoch\s+(\d+)\s*\|\s*train loss:\s*([\d.]+)\s*\|\s*val loss:\s*([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    # plt.plot(epochs, val_losses, label="Val Loss", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
    print("Saved loss_curve.png")


def runs():
    runs = json.load(open("cleaned_runs.json"))

    wins = defaultdict(int)
    losses = defaultdict(int)

    for run in runs:
        clean_character = run['character_chosen'].replace("_", " ").title()
        if run['victory']:
            wins[clean_character] += 1
        else:
            losses[clean_character] += 1

    characters = sorted(set(wins) | set(losses))
    x = range(len(characters))

    win_counts = [wins[c] for c in characters]
    loss_counts = [losses[c] for c in characters]

    plt.bar(x, loss_counts, label='Loss', color='tomato')
    plt.bar(x, win_counts, bottom=loss_counts, label='Win', color='yellowgreen')

    plt.xticks(x, characters, rotation=15)
    plt.ylabel("Number of Runs")
    plt.legend()
    plt.tight_layout()
    plt.show()

loss()