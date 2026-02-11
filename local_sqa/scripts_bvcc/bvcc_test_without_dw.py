import csv
import time
import zlib
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

SYSTEM_IDS = [
    "sys47c67",
    "sys78aec",
    "sys83aed",
    "sys91caa",
    "sys753b8",
]

MODES_TO_RUN = ["forward", "reverse", "random"]

TARGET_SECONDS = 180.0
START_INDEX = 0
MAX_FILES = 5000

RANDOM_SEED = 1234
FALLBACK_TO_CPU = False

SCRIPT_DIR = Path(__file__).resolve().parent
PACKET_DIR = SCRIPT_DIR / "packets"
PACKET_DIR.mkdir(parents=True, exist_ok=True)


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


def mean(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


def safe_infer_global(predictor_gpu, predictor_cpu, wav: np.ndarray, tag: str):
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
            preds, _, _ = predictor_gpu(wav_batch, num_samples)

        t1 = time.time()
        cpu_after = cpu_memory_mb()

        runtime = t1 - t0
        mem = gpu_peak_mb() + max(cpu_after - cpu_before, 0.0)

        mos = float(np.asarray(preds).reshape(-1)[0])
        return True, runtime, mem, mos, "gpu"

    except RuntimeError as e:
        msg = str(e).lower()
        print(f"[{tag}] runtimeerror: {e}")

        if not FALLBACK_TO_CPU or predictor_cpu is None:
            return False, None, None, None, "fail"

        if "out of memory" in msg or "cudnn" in msg or "non-contiguous" in msg:
            print(f"[{tag}] -> cpu fallback...")
            try:
                with torch.inference_mode():
                    preds, _, _ = predictor_cpu(wav_batch, num_samples)
                mos = float(np.asarray(preds).reshape(-1)[0])
                return True, None, None, mos, "cpu"
            except Exception as e2:
                print(f"[{tag}] cpu fallback failed: {e2}")

        return False, None, None, None, "fail"


def packet_path(system_id: str) -> Path:
    return PACKET_DIR / f"packet_{system_id}_t{int(TARGET_SECONDS)}_s{START_INDEX}.csv"


def build_packet(items, system_id: str, target_seconds: float):
    filtered = [(fn, mos) for (fn, mos) in items if fn.startswith(f"{system_id}-")]
    if not filtered:
        raise ValueError(f"no items found for SYSTEM_ID={system_id}")

    start = START_INDEX % len(filtered)
    ordered = filtered[start:] + filtered[:start]

    packet = []
    total_s = 0.0

    for fn, mos_t in ordered:
        if total_s >= target_seconds or len(packet) >= MAX_FILES:
            break

        p = BVCC_WAV_ROOT / fn
        wav = load_wav(p)
        dur = len(wav) / SAMPLING_RATE

        packet.append((fn, float(mos_t), str(p), float(dur)))
        total_s += float(dur)

    return packet, float(total_s), len(filtered)


def save_packet(p: Path, system_id: str, packet, total_s: float, sys_count: int):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system_id", system_id])
        w.writerow(["target_seconds", TARGET_SECONDS])
        w.writerow(["start_index", START_INDEX])
        w.writerow(["sys_count", sys_count])
        w.writerow(["packet_len", len(packet)])
        w.writerow(["packet_total_s", total_s])
        w.writerow([])
        w.writerow(["fn", "mos_target", "path", "dur_s"])
        for fn, mos_t, path_str, dur in packet:
            w.writerow([fn, mos_t, path_str, dur])


def load_packet(p: Path):
    rows = p.read_text().splitlines()
    header = "fn,mos_target,path,dur_s"
    i0 = None
    for i, r in enumerate(rows):
        if r.strip() == header:
            i0 = i + 1
            break
    if i0 is None:
        raise RuntimeError(f"invalid packet file: {p}")

    packet = []
    for r in rows[i0:]:
        r = r.strip()
        if not r:
            continue
        fn, mos_t, path_str, dur = r.split(",", 3)
        packet.append((fn, float(mos_t), Path(path_str), float(dur)))
    return packet


def order_packet(packet, mode: str, system_id: str):
    if mode == "forward":
        return packet
    if mode == "reverse":
        return list(reversed(packet))
    if mode == "random":
        h = zlib.adler32(system_id.encode("utf-8")) & 0xFFFFFFFF
        seed = (RANDOM_SEED ^ h) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(packet)).tolist()
        return [packet[i] for i in idx]
    raise ValueError(f"unknown mode: {mode}")


def out_dir_for(mode: str, system_id: str) -> Path:
    if mode == "forward":
        return SCRIPT_DIR / f"without_{system_id}"
    if mode == "reverse":
        return SCRIPT_DIR / f"without_rev_{system_id}"
    if mode == "random":
        return SCRIPT_DIR / f"without_rand_{system_id}"
    raise ValueError(mode)


def run_one(items, system_id: str, mode: str):
    p_pkt = packet_path(system_id)
    if not p_pkt.exists():
        pkt_raw, total_s, sys_count = build_packet(items, system_id, TARGET_SECONDS)
        save_packet(p_pkt, system_id, pkt_raw, total_s, sys_count)
        print(f"[{system_id}] created packet: {p_pkt.name}")
    else:
        print(f"[{system_id}] using packet:   {p_pkt.name}")

    packet = load_packet(p_pkt)
    packet = order_packet(packet, mode, system_id)

    out_dir = out_dir_for(mode, system_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if FALLBACK_TO_CPU and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=MODEL_DIR,
            checkpoint_name=CHECKPOINT_NAME,
            return_numpy=True,
            device="cpu",
        )

    print(f"\n=== {mode.upper()} | {system_id} (NO DURATION WEIGHT) ===")
    print(f"packet_len={len(packet)}  first={packet[0][0]}  last={packet[-1][0]}")
    print(f"out={out_dir}")

    elapsed = 0.0
    pred_list, tgt_list = [], []
    global_rm_rows = []

    for i, (fn, mos_t, p, dur) in enumerate(packet, start=1):
        wav = load_wav(p)
        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, wav, f"{system_id}:{mode}:seg{i}"
        )
        if not ok:
            print(f"stopping at segment {i} (running mean)")
            break

        pred_list.append(float(mos_pred))
        tgt_list.append(float(mos_t))
        elapsed += float(dur)

        pred_rm = mean(pred_list)
        tgt_rm = mean(tgt_list)
        global_rm_rows.append([elapsed, pred_rm, tgt_rm])

    if len(global_rm_rows) == 0:
        print("no segments inferred (running mean) -> skip writing")
        return

    concat_wavs = []
    concat_secs = 0.0
    global_concat_rows = []
    tgt_prefix = []

    for n, (fn, mos_t, p, dur) in enumerate(packet, start=1):
        wav = load_wav(p)
        concat_wavs.append(wav)
        concat_secs += float(dur)

        tgt_prefix.append(float(mos_t))
        wav_cat = np.concatenate(concat_wavs).astype(np.float32)

        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, wav_cat, f"{system_id}:{mode}:concat{n}"
        )
        if not ok:
            print(f"stopping at n={n} (concat)")
            break

        tgt_concat = mean(tgt_prefix)
        global_concat_rows.append([concat_secs, float(mos_pred), float(tgt_concat)])

    if len(global_concat_rows) == 0:
        print("concat failed entirely -> skip writing")
        return

    p_global_concat = out_dir / "global_mos_concat.csv"
    p_global_rm = out_dir / "global_mos_running_mean.csv"

    with open(p_global_concat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "mos_pred_concat", "mos_target_concat_mean"])
        for sec, mp, mt in global_concat_rows:
            w.writerow([round(float(sec), 6), round(float(mp), 6), round(float(mt), 6)])

    with open(p_global_rm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "mos_pred_rm_mean", "mos_target_rm_mean"])
        for t, mp, mt in global_rm_rows:
            w.writerow([round(float(t), 6), round(float(mp), 6), round(float(mt), 6)])

    print("done:")
    print(p_global_concat)
    print(p_global_rm)


def run_all():
    items = parse_bvcc_list(BVCC_TEST_LIST)
    for system_id in SYSTEM_IDS:
        for mode in MODES_TO_RUN:
            try:
                run_one(items, system_id, mode)
            except Exception as e:
                print(f"\n[ERROR] {system_id} / {mode}: {e}\n")


if __name__ == "__main__":
    run_all()
sw