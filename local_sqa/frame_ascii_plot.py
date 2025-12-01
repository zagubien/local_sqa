import sys
import numpy as np
import plotext as plt
from pathlib import Path


def ascii_frame_plot(frame_file: Path, time_file: Path):
    """Print frame MOS curve in terminal"""

    frame_file = Path(frame_file)
    time_file = Path(time_file)

    if not frame_file.exists():
        print(f"[ERROR] frame file not found: {frame_file}")
        sys.exit(1)

    if not time_file.exists():
        print(f"[ERROR] time file not found: {time_file}")
        sys.exit(1)

    # Load data
    frame = np.load(frame_file)
    time = np.load(time_file)

    if frame.shape[0] != time.shape[0]:
        print("[ERROR] frame and time arrays have different lengths!")
        sys.exit(1)

    plt.clf()            
    plt.plot(time, frame)
    plt.xlabel("Time / s")
    plt.ylabel("Predicted MOS")
    plt.title(frame_file.stem)
    plt.grid(True)
    plt.show()           

def main():
    if len(sys.argv) != 2:
        print("Usage: python frame_ascii_plot.py <basename>")
        print("Example:")
        print("  python frame_ascii_plot.py real_238.12s")
        sys.exit(0)

    base = sys.argv[1]
    base_dir = Path("/net/vol/zigor/memory_sweep_results/frame_preds")

    frame_path = base_dir / f"{base}_frame.npy"
    time_path = base_dir / f"{base}_time.npy"

    ascii_frame_plot(frame_path, time_path)


if __name__ == "__main__":
    main()
