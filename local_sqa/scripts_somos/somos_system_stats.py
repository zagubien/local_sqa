import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


AUDIO_ROOT = Path("/net/db/somos/audios")
MOS_LIST = Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt")

TARGET_SECONDS = 180.0
MIN_UTTS = 20


def wav_duration_s(p: Path) -> float:
    with wave.open(str(p), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
    return float(n) / float(sr)


def parse_system_id(utterance_filename: str) -> str:
    name = utterance_filename.strip()
    stem = name[:-4] if name.lower().endswith(".wav") else name
    if "_" not in stem:
        raise ValueError(f"cannot parse systemId (no '_' found) from: {name}")
    return stem.rsplit("_", 1)[-1]


def read_mos_list(path: Path) -> List[Tuple[str, float]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            uid, mos = ln.split(",")
            items.append((uid.strip(), float(mos)))
    return items


@dataclass
class SysStats:
    system_id: str
    n: int
    total_s: float
    mean_s: float
    p50_s: float
    p90_s: float


def main():
    if not MOS_LIST.exists():
        raise FileNotFoundError(f"missing MOS_LIST: {MOS_LIST}")

    items = read_mos_list(MOS_LIST)

    dur_by_sys: Dict[str, List[float]] = {}
    missing = 0

    for uid, mos in items:
        sys = parse_system_id(uid)
        wav = AUDIO_ROOT / uid  
        if not wav.exists():
            missing += 1
            continue
        d = wav_duration_s(wav)
        dur_by_sys.setdefault(sys, []).append(d)

    stats: List[SysStats] = []
    for sys, durs in dur_by_sys.items():
        arr = np.asarray(durs, dtype=np.float64)
        stats.append(
            SysStats(
                system_id=sys,
                n=int(arr.size),
                total_s=float(arr.sum()),
                mean_s=float(arr.mean()) if arr.size else float("nan"),
                p50_s=float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
                p90_s=float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
            )
        )

    stats.sort(key=lambda s: (s.total_s, s.n), reverse=True)

    print(f"utterances in mos list: {len(items)}")
    print(f"systems with audio found: {len(stats)}")
    print(f"missing audio files: {missing}")
    print()

    print("Top systems by total duration:")
    print("system  n   total_s   mean_s   p50_s   p90_s")
    for s in stats[:25]:
        print(f"{s.system_id:>6}  {s.n:>4}  {s.total_s:>7.1f}  {s.mean_s:>6.2f}  {s.p50_s:>6.2f}  {s.p90_s:>6.2f}")

    print()
    print(f"Candidates for TARGET_SECONDS={TARGET_SECONDS:.1f}s (and n>={MIN_UTTS}):")
    for s in stats:
        if s.total_s >= TARGET_SECONDS and s.n >= MIN_UTTS:
            print(f"  system {s.system_id}  total={s.total_s:.1f}s  n={s.n}  p90={s.p90_s:.2f}s")

    SCRIPT_DIR = Path(__file__).resolve().parent
    out_csv = SCRIPT_DIR / "system_duration_stats.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system_id", "n", "total_s", "mean_s", "p50_s", "p90_s"])
        for s in stats:
            w.writerow([s.system_id, s.n, f"{s.total_s:.6f}", f"{s.mean_s:.6f}", f"{s.p50_s:.6f}", f"{s.p90_s:.6f}"])

    print()
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
