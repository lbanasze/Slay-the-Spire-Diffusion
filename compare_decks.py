"""
compare_decks.py

Compare diffusion-generated decks to winning decks from cleaned_runs.json.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHAR_MAP = {
    "defect": "DEFECT",
    "ironclad": "IRONCLAD",
    "silent": "THE_SILENT",
    "watcher": "WATCHER",
}

def normalize(card: str) -> str:
    """Strip upgrade suffix (+1) for card-name comparison."""
    return re.sub(r"\+\d+$", "", card.strip())


def load_winning_decks(runs_path):
    """Return {CHARACTER: [[normalized card, ...], ...]} for victories only."""
    with open(runs_path) as f:
        runs = json.load(f)

    wins: dict[str, list[list[str]]] = {}
    for run in runs:
        if not run.get("victory"):
            continue
        char = run.get("character_chosen", "").upper()
        deck = [normalize(c) for c in run.get("master_deck", [])]
        wins.setdefault(char, []).append(deck)
    return wins


def load_generated_decks(trials_dir: Path) -> dict[str, dict[str, list[list[str]]]]:
    """
    Return {char_key: {"weighted": [...], "unweighted": [...]}}
    where char_key is e.g. "defect".
    """
    result: dict[str, dict[str, list[list[str]]]] = {}
    for char_dir in sorted(trials_dir.iterdir()):
        if not char_dir.is_dir():
            continue
        char = char_dir.name.lower()
        result[char] = {}
        for variant in ("weighted", "unweighted"):
            path = char_dir / f"{variant}_generated_decks.json"
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
            result[char][variant] = [[normalize(c) for c in deck] for deck in raw]
    return result


def card_frequencies(decks: list[list[str]]) -> Counter:
    counter: Counter = Counter()
    for deck in decks:
        counter.update(deck)
    return counter


def deck_sizes(decks: list[list[str]]) -> list[int]:
    return [len(d) for d in decks]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def avg_best_jaccard(generated: list[list[str]], real_wins: list[list[str]]) -> float:
    """For each generated deck, find its best Jaccard match among real wins."""
    if not generated or not real_wins:
        return 0.0
    scores = []
    for gen in generated:
        best = max(jaccard(gen, win) for win in real_wins)
        scores.append(best)
    return float(np.mean(scores))


def top_n_overlap(
    generated: list[list[str]],
    real_wins: list[list[str]],
    n: int = 20,
) -> float:
    """Fraction of the top-N real-win cards that appear in ANY generated deck."""
    top_real = {card for card, _ in card_frequencies(real_wins).most_common(n)}
    gen_cards = {card for deck in generated for card in deck}
    if not top_real:
        return 0.0
    return len(top_real & gen_cards) / len(top_real)


def collect_stats(
    char_key: str,
    variant: str,
    generated: list[list[str]],
    real_wins: list[list[str]],
    top_n: int = 10,
) -> dict:
    real_sizes = deck_sizes(real_wins)
    gen_sizes = deck_sizes(generated)
    real_freq = card_frequencies(real_wins).most_common(top_n)
    gen_freq = card_frequencies(generated).most_common(top_n)
    real_top_set = {c for c, _ in real_freq}
    gen_top_set = {c for c, _ in gen_freq}
    return {
        "char": CHAR_MAP.get(char_key, char_key.upper()),
        "variant": variant,
        "real_wins": len(real_wins),
        "gen_decks": len(generated),
        "real_mean_size": np.mean(real_sizes),
        "gen_mean_size": np.mean(gen_sizes) if gen_sizes else 0,
        "jaccard": avg_best_jaccard(generated, real_wins),
        "overlap": top_n_overlap(generated, real_wins, top_n),
        "shared": len(real_top_set & gen_top_set),
        "top_n": top_n,
        "real_freq": real_freq,
        "gen_freq": gen_freq,
        "real_top_set": real_top_set,
        "gen_top_set": gen_top_set,
    }


def print_tables(all_stats: list[dict]) -> None:
    top_n = all_stats[0]["top_n"]

    # --- Tables 1a/1b: summary stats split by variant ---
    h = f"{'Character':<12} {'Real wins':>10} {'Gen decks':>10} {'Size(real)':>10} {'Size(gen)':>9} {'Jaccard':>8} {'Overlap':>8} {f'Top-{top_n} shared':>12}"
    sep = "-" * len(h)

    for variant in ("unweighted", "weighted"):
        rows = [s for s in all_stats if s["variant"] == variant]
        if not rows:
            continue
        print(f"\n{variant.upper()}")
        print(sep)
        print(h)
        print(sep)
        for s in rows:
            print(
                f"{s['char']:<12} {s['real_wins']:>10,} {s['gen_decks']:>10} "
                f"{s['real_mean_size']:>10.1f} {s['gen_mean_size']:>9.1f} "
                f"{s['jaccard']:>8.3f} {s['overlap']:>7.1%} "
                f"{s['shared']:>6}/{top_n:<5}"
            )
        print(sep)

    # --- Table 2: top-N cards per character/variant ---
    col = 28
    for s in all_stats:
        label = f"  {s['char']} | {s['variant']} — top-{top_n} cards"
        header = f"  {'#':<4} {'Real-win card':<{col}} {'Generated card':<{col}} {'In both?':>8}"
        div = "  " + "-" * (len(header) - 2)
        print(f"\n{label}")
        print(header)
        print(div)
        for i, ((rc, _), (gc, _)) in enumerate(zip(s["real_freq"], s["gen_freq"]), 1):
            mark = "✓" if rc in s["gen_top_set"] else ""
            print(f"  {i:<4} {rc:<{col}} {gc:<{col}} {mark:>8}")
        print(div)


def plot_card_frequencies(
    char_key: str,
    variant: str,
    generated: list[list[str]],
    real_wins: list[list[str]],
    top_n: int = 20,
    out_dir: Path = Path("."),
) -> None:
    real_freq = card_frequencies(real_wins)
    gen_freq = card_frequencies(generated)

    # Union of top-N from both
    top_real = {c for c, _ in real_freq.most_common(top_n)}
    top_gen = {c for c, _ in gen_freq.most_common(top_n)}
    cards = sorted(top_real | top_gen)

    real_counts = np.array([real_freq[c] for c in cards], dtype=float)
    gen_counts = np.array([gen_freq[c] for c in cards], dtype=float)

    # Normalize to fractions (per-deck occurrence rate)
    if real_wins:
        real_counts /= len(real_wins)
    if generated:
        gen_counts /= len(generated)

    x = np.arange(len(cards))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(cards) * 0.6), 6))
    ax.bar(x - width / 2, real_counts, width, label="Real wins", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, gen_counts, width, label=f"Generated ({variant})", color="tomato", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cards, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg cards per deck")
    char_label = CHAR_MAP.get(char_key, char_key.upper())
    ax.set_title(f"{char_label} — Card frequency: real wins vs {variant} generated")
    ax.legend()
    plt.tight_layout()

    fname = out_dir / f"{char_key}_{variant}_card_freq.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_deck_sizes(
    char_key: str,
    variant: str,
    generated: list[list[str]],
    real_wins: list[list[str]],
    out_dir: Path = Path("."),
) -> None:
    real_sizes = deck_sizes(real_wins)
    gen_sizes = deck_sizes(generated)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, max(max(real_sizes, default=0), max(gen_sizes, default=0)) + 5)
    ax.hist(real_sizes, bins=bins, alpha=0.6, label="Real wins", color="steelblue", density=True)
    ax.hist(gen_sizes, bins=bins, alpha=0.6, label=f"Generated ({variant})", color="tomato", density=True)

    ax.set_xlabel("Deck size")
    ax.set_ylabel("Density")
    char_label = CHAR_MAP.get(char_key, char_key.upper())
    ax.set_title(f"{char_label} — Deck size distribution: real wins vs {variant} generated")
    ax.legend()
    plt.tight_layout()

    fname = out_dir / f"{char_key}_{variant}_deck_size.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def main():
    runs_path = Path("cleaned_runs.json")
    trials_dir = Path("trials")
    out_dir = Path("comparison_plots")

    print(f"Loading winning runs from {runs_path} ...")
    winning_decks = load_winning_decks(runs_path)
    for char, decks in winning_decks.items():
        print(f"  {char}: {len(decks):,} wins")

    print(f"\nLoading generated decks from {trials_dir} ...")
    generated_all = load_generated_decks(trials_dir)

    all_stats = []
    for char_key, variants in sorted(generated_all.items()):
        char_upper = CHAR_MAP.get(char_key, char_key.upper())
        real_wins = winning_decks.get(char_upper, [])
        if not real_wins:
            print(f"No real winning decks found for {char_upper}, skipping.")
            continue

        for variant, generated in sorted(variants.items()):
            if not generated:
                print(f"No generated decks for {char_key}/{variant}, skipping.")
                continue

            all_stats.append(collect_stats(char_key, variant, generated, real_wins, top_n=10))

    print_tables(all_stats)


if __name__ == "__main__":
    main()
