import csv
from pathlib import Path

import numpy as np
import torch

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio



MODEL_DIR = "/net/vol/zigor/checkpoints/27"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/49/121052")  # male
#CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/47/122796")   # female

OUTPUT_ROOT = Path("/net/vol/zigor/single_chapter_running_mean/")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PREFIX = "results_running_mean"



# AUDIO LOADER
AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)



# HELPERS
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
        assert audio.ndim == 1

        audio_list.append(audio)
        durations.append(len(audio) / SAMPLING_RATE)

    return files, audio_list, durations


def safe_infer(predictor: SpeechQualityPredictor, wav: np.ndarray):
    wav = np.asarray(wav, dtype=np.float32)
    wav_batch = wav[None, :]
    num_samples = [wav.shape[0]]

    try:
        with torch.inference_mode():
            preds, _, _ = predictor(wav_batch, num_samples)
        return True, float(preds[0])
    except Exception as e:
        print(f"Inference failed: {e}")
        return False, None


def get_output_dir() -> Path:
    parent_id = CHAPTER_DIR.parent.name
    chapter_id = CHAPTER_DIR.name
    out_dir = OUTPUT_ROOT / f"{parent_id}_{chapter_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_csv_path() -> Path:
    out_dir = get_output_dir()
    run_name = Path(MODEL_DIR).name
    csv_path = out_dir / f"{CSV_PREFIX}_{run_name}.csv"

    if not csv_path.exists():
        return csv_path

    idx = 1
    while True:
        candidate = out_dir / f"{CSV_PREFIX}_{run_name}_v{idx}.csv"
        if not candidate.exists():
            return candidate
        idx += 1



# MAIN SWEEP
def run_running_mean_sweep():
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

    running_sum = 0.0
    elapsed_seconds = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sentence_idx",
                "sentence_duration_s",
                "elapsed_s",
                "mos_sentence",
                "mos_running_mean",
            ]
        )

        for idx, (wav, dur) in enumerate(zip(audio_list, durations), start=1):
            print(f"Inferencing sentence {idx} ({dur:.2f} s)")

            success, mos = safe_infer(predictor, wav)
            if not success:
                print(f"Stopping at sentence {idx}")
                break

            running_sum += mos
            elapsed_seconds += dur
            mos_mean = running_sum / idx

            writer.writerow(
                [
                    idx,
                    round(dur, 3),
                    round(elapsed_seconds, 3),
                    round(mos, 4),
                    round(mos_mean, 4),
                ]
            )

    print(f"\nDONE → {csv_path}")


if __name__ == "__main__":
    run_running_mean_sweep()
