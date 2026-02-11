import csv
from pathlib import Path

import numpy as np
import torch

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


MODEL_DIR = "/net/vol/zigor/checkpoints/27"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

MALE_CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/49/121052")
FEMALE_CHAPTER_DIR = Path("/net/db/LibriSpeech/train-other-500/47/122796")

OUTPUT_ROOT = Path("/net/vol/zigor/two_speaker_running_mean/")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PREFIX = "results_running_mean"

AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


def load_sentences(chapter_dir: Path, speaker_tag: str):
    files = sorted(chapter_dir.glob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found in {chapter_dir}")

    segments = []
    for f in files:
        ex = {"audio_path": {"observation": str(f)}}
        ex = AUDIO_LOADER(ex)
        wav = ex["audio"].astype(np.float32)
        if wav.ndim != 1:
            raise ValueError(f"Expected 1D audio, got {wav.shape} for {f}")

        dur = len(wav) / SAMPLING_RATE
        segments.append((wav, dur, speaker_tag, f.name))

    return segments


def build_interleaved_segments(male_segments, female_segments):
    n_pairs = min(len(male_segments), len(female_segments))
    interleaved = []
    for i in range(n_pairs):
        interleaved.append(male_segments[i])
        interleaved.append(female_segments[i])
    return interleaved


def safe_infer(predictor: SpeechQualityPredictor, wav: np.ndarray):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got {wav.shape}")

    wav_batch = wav[None, :]
    num_samples = [int(wav.shape[0])]

    try:
        with torch.inference_mode():
            preds, _, _ = predictor(wav_batch, num_samples)
        return True, float(preds[0])
    except Exception as e:
        print(f"Inference failed: {e}")
        return False, None


def get_experiment_output_dir() -> Path:
    male_parent = MALE_CHAPTER_DIR.parent.name
    male_id = MALE_CHAPTER_DIR.name
    female_parent = FEMALE_CHAPTER_DIR.parent.name
    female_id = FEMALE_CHAPTER_DIR.name

    folder_name = f"{male_parent}_{male_id}__{female_parent}_{female_id}"
    out_dir = OUTPUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_csv_path() -> Path:
    out_dir = get_experiment_output_dir()
    run_name = Path(MODEL_DIR).name
    base = out_dir / f"{CSV_PREFIX}_{run_name}.csv"

    if not base.exists():
        return base

    idx = 1
    while True:
        cand = out_dir / f"{CSV_PREFIX}_{run_name}_v{idx}.csv"
        if not cand.exists():
            return cand
        idx += 1


def run_two_speaker_running_mean():
    print(f"Loading male chapter:   {MALE_CHAPTER_DIR}")
    male_segments = load_sentences(MALE_CHAPTER_DIR, "M")
    print(f"Male sentences: {len(male_segments)}")

    print(f"Loading female chapter: {FEMALE_CHAPTER_DIR}")
    female_segments = load_sentences(FEMALE_CHAPTER_DIR, "F")
    print(f"Female sentences: {len(female_segments)}")

    segments = build_interleaved_segments(male_segments, female_segments)
    print(f"Interleaved segments: {len(segments)} (M1,F1,M2,F2,...)")

    print(f"Loading model: {MODEL_DIR} / {CHECKPOINT_NAME}")
    predictor = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    csv_path = get_csv_path()
    print(f"Writing results to: {csv_path}")

    running_sum = 0.0
    elapsed_seconds = 0.0

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "segment_idx",
                "speaker",
                "file",
                "segment_duration_s",
                "elapsed_s",
                "mos_segment",
                "mos_running_mean",
            ]
        )

        for idx, (wav, dur, spk, fname) in enumerate(segments, start=1):
            print(f"Inferencing segment {idx} ({spk}, {dur:.2f} s)")

            success, mos = safe_infer(predictor, wav)
            if not success:
                print(f"Stopping at segment {idx}")
                break

            running_sum += mos
            elapsed_seconds += dur
            mos_mean = running_sum / idx

            w.writerow(
                [
                    idx,
                    spk,
                    fname,
                    round(dur, 3),
                    round(elapsed_seconds, 3),
                    round(mos, 4),
                    round(mos_mean, 4),
                ]
            )

    print(f"DONE -> {csv_path}")


if __name__ == "__main__":
    run_two_speaker_running_mean()
