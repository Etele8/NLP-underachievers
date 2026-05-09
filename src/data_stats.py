from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from .data import load_split_csv, summarize_examples


# PATH SETUP 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data" / "raw"

TRAIN_PATH = DATA_DIR / "ner_spaeng_train.csv"
VAL_PATH = DATA_DIR / "ner_spaeng_validation.csv"
TEST_PATH = DATA_DIR / "ner_spaeng_test.csv"



# STATISTICS

def build_dataset_stats(train, val, test):
    train_stats = summarize_examples(train)
    val_stats = summarize_examples(val)
    test_stats = summarize_examples(test)

    df = pd.DataFrame([
        {
            "Split": "Train",
            "Examples": train_stats["num_examples"],
            "Avg Tokens": train_stats["mean_tokens_per_example"],
            "Max Tokens": train_stats["max_tokens_per_example"],
        },
        {
            "Split": "Validation",
            "Examples": val_stats["num_examples"],
            "Avg Tokens": val_stats["mean_tokens_per_example"],
            "Max Tokens": val_stats["max_tokens_per_example"],
        },
        {
            "Split": "Test",
            "Examples": test_stats["num_examples"],
            "Avg Tokens": test_stats["mean_tokens_per_example"],
            "Max Tokens": test_stats["max_tokens_per_example"],
        },
    ])

    return df, train_stats, val_stats, test_stats



# PLOTS

def plot_token_hist(train, val, test):
    def lengths(examples):
        return [len(ex["tokens"]) for ex in examples]

    plt.figure()
    plt.hist(lengths(train), bins=30, alpha=0.6, label="Train")
    plt.hist(lengths(val), bins=30, alpha=0.6, label="Validation")
    plt.hist(lengths(test), bins=30, alpha=0.6, label="Test")
    plt.title("Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_label_frequency(train_stats):
    labels = train_stats["label_distribution"]

    plt.figure()
    plt.bar(labels.keys(), labels.values())
    plt.title("Label Frequency (Train)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


def plot_entity_distribution(train_stats):
    entities = train_stats["entity_type_counts"]

    plt.figure()
    plt.bar(entities.keys(), entities.values())
    plt.title("Entity Distribution (Train)")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# MAIN FUNCTION

def run_dataset_analysis():
    print(f"Loading data from: {DATA_DIR}")

    train = load_split_csv(str(TRAIN_PATH), "train")
    val = load_split_csv(str(VAL_PATH), "validation")
    test = load_split_csv(str(TEST_PATH), "test")

    # Stats table
    table, train_stats, val_stats, test_stats = build_dataset_stats(train, val, test)

    print("\n===== DATASET SUMMARY TABLE =====")
    print(table)

    # Plots
    plot_token_hist(train, val, test)
    plot_label_frequency(train_stats)
    plot_entity_distribution(train_stats)

    return table


# CLI ENTRY

if __name__ == "__main__":
    run_dataset_analysis()