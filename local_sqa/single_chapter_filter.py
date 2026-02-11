import csv
import time
from pathlib import Path

import numpy as np
import torch
import psutil

from scipy.signal import butter, filtfilt   

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio

MODEL_DIR = "/net/vol/zigor/checkpoints/27"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

# CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/49/121052") #male
CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/47/122796") #female
                   
OUTPUT_ROOT = Path("/net/vol/zigor/single_chapter_filter/")  
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PREFIX = "results"


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
        assert audio.ndim == 1, f"Expected 1D audio, got {audio.shape} for {f}"

        audio_list.append(audio)
        durations.append(len(audio) / SAMPLING_RATE)

    return files, audio_list, durations


def concat_audio(audio_list, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(audio_list[:n]).astype(np.float32)



# Noise + Bandpass

def add_noise(wav: np.ndarray, snr_db: float = 20.0) -> np.ndarray:   
    """Add white noise at a given SNR level."""
    rms = np.sqrt(np.mean(wav**2))
    noise_rms = rms / (10**(snr_db / 20))
    noise = np.random.normal(0, noise_rms, wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def bandpass_filter(wav: np.ndarray, fs: int, low: float = 500.0, high: float = 2500.0) -> np.ndarray:  
    """Apply a Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(N=4, Wn=[low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, wav).astype(np.float32)





def safe_infer(predictor: SpeechQualityPredictor, wav: np.ndarray, tag: str):
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
            preds, frame_preds, seq_len = predictor(wav_batch, num_samples)

        t1 = time.time()
        cpu_after = cpu_memory_mb()

        runtime = t1 - t0
        mem = gpu_peak_mb() + max(cpu_after - cpu_before, 0.0)

        mos = float(preds[0])
        return True, runtime, mem, mos

    except RuntimeError as e:
        msg = str(e).lower()
        print(f"[{tag}] RuntimeError on GPU: {e}")

        if "cudnn" in msg or "non-contiguous" in msg:
            print(f"[{tag}] → CPU fallback…")
            try:
                with torch.inference_mode():
                    predictor_cpu = SpeechQualityPredictor(
                        ssl_mos=predictor.model.to("cpu"),
                        prepare_example_before=predictor.prepare_example_before,
                        device="cpu",
                        return_numpy=predictor.return_numpy,
                        median_filter_size=predictor.median_filter_size,
                    )
                    preds, frame_preds, seq_len = predictor_cpu(
                        wav_batch, num_samples
                    )
                mos = float(preds[0])
                return True, None, None, mos
            except Exception as e2:
                print(f"[{tag}] CPU fallback also failed: {e2}")

        return False, None, None, None


def get_chapter_output_dir() -> Path:
    parent_id = CHAPTER_DIR.parent.name
    chapter_id = CHAPTER_DIR.name
    folder_name = f"{parent_id}_{chapter_id}"

    
    chapter_dir = OUTPUT_ROOT / folder_name   
    chapter_dir.mkdir(parents=True, exist_ok=True)
    return chapter_dir


def get_csv_path() -> Path:
    chapter_out = get_chapter_output_dir()
    run_name = Path(MODEL_DIR).name
    base_prefix = f"{CSV_PREFIX}_{run_name}"
    csv_path = chapter_out / f"{base_prefix}.csv"

    if not csv_path.exists():
        return csv_path

    idx = 1
    while True:
        candidate = chapter_out / f"{base_prefix}_v{idx}.csv"
        if not candidate.exists():
            return candidate
        idx += 1


def run_sweep():
    print(f"Loading chapter from: {CHAPTER_DIR}")
    files, audio_list, durations = load_sentences(CHAPTER_DIR)
    print(f"Found sentences: {len(files)}")
    print(f"Total duration: {sum(durations):.2f} s")

    print(f"\nLoading model from: {MODEL_DIR}")
    predictor = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    csv_path = get_csv_path()
    print(f"\nWriting results to: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        
        writer.writerow([
            "sentences", "seconds",
            "success_clean", "runtime_clean_s", "memory_clean_mb", "mos_clean",
            "success_noise", "runtime_noise_s", "memory_noise_mb", "mos_noise",
            "success_bp", "runtime_bp_s", "memory_bp_mb", "mos_bp",
        ])

        for n in range(1, len(audio_list) + 1):
            wav_clean = concat_audio(audio_list, n)
            total_len = len(wav_clean) / SAMPLING_RATE

            print(f"\nTesting first {n} sentences → {total_len:.2f} s")

            # CLEAN
            success_c, runtime_c, mem_c, mos_c = safe_infer(predictor, wav_clean, f"clean_{n}")

            # NOISE
            wav_noise = add_noise(wav_clean, snr_db=20)
            success_n, runtime_n, mem_n, mos_n = safe_infer(predictor, wav_noise, f"noise_{n}")

            # BANDPASS
            wav_bp = bandpass_filter(wav_clean, SAMPLING_RATE)
            success_b, runtime_b, mem_b, mos_b = safe_infer(predictor, wav_bp, f"bp_{n}")

            writer.writerow([
                n, round(total_len, 3),
                success_c, None if runtime_c is None else round(runtime_c, 4),
                None if mem_c is None else round(mem_c, 4), mos_c,
                success_n, None if runtime_n is None else round(runtime_n, 4),
                None if mem_n is None else round(mem_n, 4), mos_n,
                success_b, None if runtime_b is None else round(runtime_b, 4),
                None if mem_b is None else round(mem_b, 4), mos_b,
            ])

            if (not success_c) or (not success_n) or (not success_b):
                print(f"Aborting at n={n} due to failure.")
                break

    print(f"\nDONE → {csv_path}")


if __name__ == "__main__":
    run_sweep()
