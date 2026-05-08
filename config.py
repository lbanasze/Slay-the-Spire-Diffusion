BATCH_SIZE = 128
CHARACTERS = {"IRONCLAD": 0, "THE_SILENT": 1, "DEFECT": 2, "WATCHER": 3}
D_MODEL = 128 # Size of vector for each token
DROPOUT = 0.1 # Prevents overfitting
EPOCHS = 25
KEEP_FIELDS = [
    "floor_reached", "is_ascension_mode", "master_deck",
    "relics", "character_chosen", "is_daily", "victory", "killed_by"
]
LR = 3e-4
MAX_DECK_SIZE = 40
N_HEADS = 4 # Attention heads
N_LAYERS = 3
N_SAMPLES = 8
T = 10
WIN_CONDITION = True
WEIGHTED_SAMPLING = True
