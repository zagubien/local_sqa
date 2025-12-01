import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os

# load Karla font  
FONT_PATH = "/Users/igorzagubien/Library/Fonts/Karla-VariableFont_wght.ttf"
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError("Karla font not found at: " + FONT_PATH)
PROP = fm.FontProperties(fname=FONT_PATH)

# UPB colors
UPB_BLUE = "#0A75C4"
UPB_DARK = "#0025AA"
UPB_PINK = "#EF3A84"

# NOCH ÄNDERN 
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "memory_sweep_results.csv"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find CSV file: {CSV_PATH}")

    print(f"Loading CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # basic cleaning
    df = df[df["success"] == True]

    lengths = df["length_seconds"].values
    runtime = df["runtime_s"].values
    memory = df["memory_mb"].values
    mos = df["global_mos"].values

    # Plot 1: Runtime
    plt.figure(figsize=(12, 6))
    plt.plot(lengths, runtime, marker="o", color=UPB_BLUE, linewidth=2)
    plt.title("Runtime vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("runtime [s]", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.xticks(fontproperties=PROP, fontsize=18)
    plt.yticks(fontproperties=PROP, fontsize=18)
    plt.tight_layout()
    plt.show()

    # Plot 2: Memory
    plt.figure(figsize=(12, 6))
    plt.plot(lengths, memory, marker="o", color=UPB_DARK, linewidth=2)
    plt.title("Peak Memory vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("memory [MB]", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.xticks(fontproperties=PROP, fontsize=18)
    plt.yticks(fontproperties=PROP, fontsize=18)
    plt.tight_layout()
    plt.show()

    # Plot 3: MOS
    plt.figure(figsize=(12, 6))
    plt.plot(lengths, mos, marker="o", color=UPB_PINK, linewidth=2)
    plt.title("MOS vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("Predicted MOS", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.xticks(fontproperties=PROP, fontsize=18)
    plt.yticks(fontproperties=PROP, fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
