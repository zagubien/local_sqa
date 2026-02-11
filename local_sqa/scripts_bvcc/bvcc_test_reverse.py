import csv
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio



MODEL_DIR = "/net/vol/zigor/checkpoints/24"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

BVCC_TEST_LIST = Path("/net/db/BVCC/main/DATA/sets/test_mos_list.txt")
BVCC_WAV_ROOT = Path("/net/db/BVCC/main/DATA/wav")


SYSTEM_ID = "sys47c67"
#SYSTEM_ID = "sys91caa"
#SYSTEM_ID = "sys83aed"
#SYSTEM_ID = "sys78aec"
#SYSTEM_ID = "sys753b8"
TARGET_SECONDS = 180.0
START_INDEX = 0
MAX_FILES = 5000

SCRIPT_DIR = Path(__file__).resolve().parent

# output folder: rev_sysXXXX
OUT_DIR = SCRIPT_DIR / f"rev_{SYSTEM_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def gpu_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)



AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


def parse_bvcc_list(list_path: Path):
    items = []
    for ln in list_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        fn, mos = ln.split(",")
        items.append((fn.strip(), float(mos)))
    return items


def load_wav(path: Path) -> np.ndarray:
    ex = {"audio_path": {"observation": str(path)}}
    ex = AUDIO_LOADER(ex)
    wav = ex["audio"].astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D audio, got {wav.shape} for {path}")
    return wav


def durw_mean(values, durations) -> float:
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    w = np.maximum(d, 1e-12)
    return float(np.sum(v * w) / np.sum(w))


def safe_infer_global(predictor: SpeechQualityPredictor, wav: np.ndarray, tag: str):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D waveform, got {wav.shape}")

    wav_batch = wav[None, :]
    num_samples = [int(wav.shape[0])]

    cpu_before = cpu_memory_mb()
    gpu_reset()
    t0 = time.time()

    try:
        with torch.inference_mode():
            preds, _, _ = predictor(wav_batch, num_samples)

        t1 = time.time()
        cpu_after = cpu_memory_mb()

        runtime = t1 - t0
        mem = gpu_peak_mb() + max(cpu_after - cpu_before, 0.0)

        mos = float(np.asarray(preds).reshape(-1)[0])
        return True, runtime, mem, mos

    except RuntimeError as e:
        msg = str(e).lower()
        print(f"[{tag}] runtimeerror: {e}")

        if "cudnn" in msg or "non-contiguous" in msg or "out of memory" in msg:
            print(f"[{tag}] -> cpu fallback...")
            try:
                with torch.inference_mode():
                    predictor_cpu = SpeechQualityPredictor(
                        ssl_mos=predictor.model.to("cpu"),
                        prepare_example_before=predictor.prepare_example_before,
                        device="cpu",
                        return_numpy=predictor.return_numpy,
                        median_filter_size=predictor.median_filter_size,
                    )
                    preds, _, _ = predictor_cpu(wav_batch, num_samples)

                mos = float(np.asarray(preds).reshape(-1)[0])
                return True, None, None, mos
            except Exception as e2:
                print(f"[{tag}] cpu fallback failed: {e2}")

        return False, None, None, None


def pick_system_sequence(items, target_seconds: float):
    # filter list entries for this system id
    filtered = [(fn, mos) for (fn, mos) in items if fn.startswith(f"{SYSTEM_ID}-")]
    if not filtered:
        raise ValueError(f"no items found for SYSTEM_ID={SYSTEM_ID}")

    # keep same "start index" behavior as forward script, then reverse after picking
    start = START_INDEX % len(filtered)
    ordered = filtered[start:] + filtered[:start]

    seq = []
    total_s = 0.0

    for fn, mos_t in ordered:
        if total_s >= target_seconds or len(seq) >= MAX_FILES:
            break

        p = BVCC_WAV_ROOT / fn
        wav = load_wav(p)
        dur = len(wav) / SAMPLING_RATE

        seq.append((fn, float(mos_t), p, wav, float(dur)))
        total_s += float(dur)

    # reverse the picked sequence 
    seq = list(reversed(seq))

    return seq, float(total_s), len(filtered)



# main

def run():
    items = parse_bvcc_list(BVCC_TEST_LIST)
    picked, total_s, sys_count = pick_system_sequence(items, TARGET_SECONDS)

    print(f"SYSTEM_ID={SYSTEM_ID} -> available files in test list: {sys_count}")
    print(f"picked files: {len(picked)}")
    print(f"picked duration: {total_s:.2f}s (target={TARGET_SECONDS:.2f}s)")
    print(f"writing to: {OUT_DIR}")
    if picked:
        print(f"first file (reversed): {picked[0][0]}")
        print(f"last  file (reversed): {picked[-1][0]}")

    predictor = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    
    # running mean 
    elapsed = 0.0
    pred_list = []
    tgt_list = []
    dur_list = []

    global_rm_rows = []  # elapsed_s, pred_rm_durw, target_rm_durw

    for i, (fn, mos_t, p, wav, dur) in enumerate(picked, start=1):
        ok, rt, mem, mos_pred = safe_infer_global(predictor, wav, f"seg{i}")
        if not ok:
            print(f"stopping at segment {i} (running mean path)")
            break

        pred_list.append(float(mos_pred))
        tgt_list.append(float(mos_t))
        dur_list.append(float(dur))
        elapsed += float(dur)

        pred_rm = durw_mean(pred_list, dur_list)
        tgt_rm = durw_mean(tgt_list, dur_list)

        global_rm_rows.append([elapsed, pred_rm, tgt_rm])

    if len(global_rm_rows) == 0:
        raise RuntimeError("no segments inferred in running mean path")

    
    # concat 
    concat_wavs = []
    concat_secs = 0.0

    global_concat_rows = []  # seconds, pred_concat, target_concat_durw
    tgt_prefix = []
    dur_prefix = []

    for n, (fn, mos_t, p, wav, dur) in enumerate(picked, start=1):
        concat_wavs.append(wav)
        concat_secs += float(dur)

        tgt_prefix.append(float(mos_t))
        dur_prefix.append(float(dur))

        wav_cat = np.concatenate(concat_wavs).astype(np.float32)

        ok, rt, mem, mos_pred = safe_infer_global(predictor, wav_cat, f"concat{n}")
        if not ok:
            print(f"stopping at n={n} (concat path)")
            break

        tgt_concat = durw_mean(tgt_prefix, dur_prefix)
        global_concat_rows.append([concat_secs, float(mos_pred), float(tgt_concat)])

    if len(global_concat_rows) == 0:
        raise RuntimeError("concat failed entirely")


    # write outputs 

    p_global_concat = OUT_DIR / "global_mos_concat.csv"
    p_global_rm = OUT_DIR / "global_mos_running_mean.csv"

    with open(p_global_concat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "mos_pred_concat", "mos_target_concat_durw"])
        for sec, mp, mt in global_concat_rows:
            w.writerow([round(float(sec), 6), round(float(mp), 6), round(float(mt), 6)])

    with open(p_global_rm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "mos_pred_rm_durw", "mos_target_rm_durw"])
        for t, mp, mt in global_rm_rows:
            w.writerow([round(float(t), 6), round(float(mp), 6), round(float(mt), 6)])

    print("\nDONE (global-only, reversed order)")
    print(p_global_concat)
    print(p_global_rm)


if __name__ == "__main__":
    run()
