import csv
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from scipy.signal import butter, filtfilt

from local_sqa.modules.data_loader import LoadAudio
from local_sqa.modules.ssl_mos import SAMPLING_RATE, SpeechQualityPredictor


MODEL_DIR = "/net/vol/zigor/checkpoints/27"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

# CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/49/121052")  # male
CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/47/122796")   # female

OUTPUT_ROOT = Path("/net/vol/zigor/single_chapter_filter_N/")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PREFIX = "results"

APPLY_EVERY_N = 4

NOISE_SNR_DB = 10.0

BP_LOW_HZ = 500.0
BP_HIGH_HZ = 2500.0
BP_ORDER = 4

RNG_SEED = 0


def cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def gpu_reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


def load_sentences(chapter_dir: Path):
    files = sorted(chapter_dir.glob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found in {chapter_dir}")

    audio_list = []
    durations = []

    for f in files:
        ex = {"audio_path": {"observation": str(f)}}
        ex = AUDIO_LOADER(ex)
        audio = ex["audio"].astype(np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio, got {audio.shape} for {f}")

        audio_list.append(audio)
        durations.append(len(audio) / SAMPLING_RATE)

    return files, audio_list, durations


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = rng.normal(0.0, noise_rms, wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def bandpass_filter(wav: np.ndarray, fs: int, low: float, high: float, order: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    nyq = fs / 2.0
    b, a = butter(N=order, Wn=[low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, wav).astype(np.float32)


def concat_variant(audio_list, n: int, mode: str, every_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros(1, dtype=np.float32)

    out = []
    for i in range(n):
        seg = audio_list[i]
        idx1 = i + 1

        if mode == "clean":
            out.append(seg)
            continue

        if every_n <= 0 or (idx1 % every_n) != 0:
            out.append(seg)
            continue

        if mode == "noise":
            out.append(add_noise(seg, NOISE_SNR_DB, rng))
        elif mode == "bp":
            out.append(bandpass_filter(seg, SAMPLING_RATE, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return np.concatenate(out).astype(np.float32)


def safe_infer_gpu_only(predictor: SpeechQualityPredictor, wav: np.ndarray, tag: str):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got {wav.shape}")

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

        mos = float(preds[0])
        return True, runtime, mem, mos

    except RuntimeError as e:
        print(f"[{tag}] GPU RuntimeError: {e}")
        return False, None, None, None


def get_chapter_output_dir() -> Path:
    parent_id = CHAPTER_DIR.parent.name
    chapter_id = CHAPTER_DIR.name
    folder_name = f"{parent_id}_{chapter_id}"

    out_dir = OUTPUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_csv_path() -> Path:
    out_dir = get_chapter_output_dir()
    run_name = Path(MODEL_DIR).name

    base_prefix = f"{CSV_PREFIX}_{run_name}_N{APPLY_EVERY_N}"
    csv_path = out_dir / f"{base_prefix}.csv"

    if not csv_path.exists():
        return csv_path

    idx = 1
    while True:
        candidate = out_dir / f"{base_prefix}_v{idx}.csv"
        if not candidate.exists():
            return candidate
        idx += 1


def run_sweep():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only and will not run on CPU.")

    print(f"Loading chapter from: {CHAPTER_DIR}")
    _, audio_list, durations = load_sentences(CHAPTER_DIR)
    print(f"Found sentences: {len(audio_list)}")
    print(f"Total duration: {sum(durations):.2f} s")
    print(f"Applying noise/bandpass only every Nth sentence: N={APPLY_EVERY_N}")

    rng = np.random.default_rng(RNG_SEED) if RNG_SEED is not None else np.random.default_rng()

    print(f"\nLoading model from: {MODEL_DIR}")
    predictor = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device="cuda",
    )

    csv_path = get_csv_path()
    print(f"\nWriting results to: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sentences", "seconds",
            "success_clean", "runtime_clean_s", "memory_clean_mb", "mos_clean",
            "success_noise", "runtime_noise_s", "memory_noise_mb", "mos_noise",
            "success_bp", "runtime_bp_s", "memory_bp_mb", "mos_bp",
        ])

        for n in range(1, len(audio_list) + 1):
            wav_clean = concat_variant(audio_list, n, "clean", APPLY_EVERY_N, rng)
            wav_noise = concat_variant(audio_list, n, "noise", APPLY_EVERY_N, rng)
            wav_bp = concat_variant(audio_list, n, "bp", APPLY_EVERY_N, rng)

            total_len = len(wav_clean) / SAMPLING_RATE
            print(f"\nTesting first {n} sentences -> {total_len:.2f} s")

            sc, rtc, memc, mosc = safe_infer_gpu_only(predictor, wav_clean, f"clean_{n}")
            sn, rtn, memn, mosn = safe_infer_gpu_only(predictor, wav_noise, f"noise_N{APPLY_EVERY_N}_{n}")
            sb, rtb, memb, mosb = safe_infer_gpu_only(predictor, wav_bp, f"bp_N{APPLY_EVERY_N}_{n}")

            w.writerow([
                n, round(total_len, 3),
                sc, None if rtc is None else round(rtc, 4), None if memc is None else round(memc, 4), mosc,
                sn, None if rtn is None else round(rtn, 4), None if memn is None else round(memn, 4), mosn,
                sb, None if rtb is None else round(rtb, 4), None if memb is None else round(memb, 4), mosb,
            ])

            if (not sc) or (not sn) or (not sb):
                print(f"Aborting at n={n} due to failure.")
                break

    print(f"\nDONE -> {csv_path}")


if __name__ == "__main__":
    run_sweep()
