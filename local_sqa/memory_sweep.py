import os
import csv
import time
from pathlib import Path
import numpy as np
import soundfile as sf

import torch
import psutil
from paderbox.io import load_audio
from paderbox.transform.module_resample import resample_sox

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE


# config
MODEL_DIR = "/net/vol/zigor/checkpoints/3"
OUTPUT_DIR = Path("/net/vol/zigor/memory_sweep_results/")
AUDIO_ROOT = Path("/net/db/librispeech_long/")
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

FRAME_DIR = OUTPUT_DIR / "frame_preds"
FRAME_DIR.mkdir(parents=True, exist_ok=True)



#memory
def get_cpu_memory_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def safe_gpu_memory_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)



#  inference 
def safe_infer(model, wav, num_samples, tag_name):

    try:
        cpu_before = get_cpu_memory_mb()
        safe_gpu_memory_reset()

        t0 = time.time()
        preds, frame_preds, seq_len = model(wav, num_samples)
        t1 = time.time()

        runtime = t1 - t0
        gpu_peak = get_gpu_peak_mb()
        cpu_after = get_cpu_memory_mb()

        mem_total = gpu_peak + max(cpu_after - cpu_before, 0)

        frame_np = frame_preds[0][:seq_len[0]]
        time_axis = np.arange(len(frame_np)) / model.model.encoder.frame_rate

        np.save(FRAME_DIR / f"{tag_name}_frame.npy", frame_np)
        np.save(FRAME_DIR / f"{tag_name}_time.npy", time_axis)

        return True, runtime, mem_total, float(preds[0])


    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM detected — stopping.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, None, None, None
        raise e


# load all librispeech-long files
def collect_real_files():
    flacs = list(AUDIO_ROOT.rglob("*.flac"))
    results = []

    for f in flacs:
        try:
            info = sf.info(str(f))
            results.append((f, float(info.duration)))
        except:
            pass

    results.sort(key=lambda x: x[1])
    return results



# concatenate audios to one long audio
def concat_audio_list(audio_list):
    if len(audio_list) == 1:
        return audio_list[0]
    return np.concatenate(audio_list, axis=0)



#main
def run_sweep():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "memory_sweep_results.csv"

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "length_seconds",
            "success",
            "runtime_s",
            "memory_mb",
            "global_mos",
            "frame_pred_file",
            "timestamp_file"
        ])

        print("Loading model...")
        model = SpeechQualityPredictor(
            storage_dir=MODEL_DIR,
            checkpoint_name=CHECKPOINT_NAME,
            return_numpy=True
        )

        files = collect_real_files()

        # progressive concatenation storage
        accumulated_audio = []
        accumulated_length = 0.0

        for idx, (path, dur) in enumerate(files):

            # load new file
            wav, sr = load_audio(path, return_sample_rate=True)
            if sr != SAMPLING_RATE:
                wav = resample_sox(wav, sr, SAMPLING_RATE)
            wav = wav.astype(np.float32)

            # append to accumulated list
            accumulated_audio.append(wav)
            accumulated_length += dur

            # build new long audio
            full_audio = concat_audio_list(accumulated_audio)
            full_audio_batch = full_audio[None]

            tag = f"concat_{round(accumulated_length,2)}s"

            print(f"Testing concatenated length {accumulated_length:.2f}s")

            success, rt, mem, mos = safe_infer(
                model,
                full_audio_batch,
                [full_audio_batch.shape[-1]],
                tag
            )

            writer.writerow([
                round(accumulated_length, 3),
                success,
                None if rt is None else round(rt, 4),
                None if mem is None else round(mem, 4),
                mos,
                f"{tag}_frame.npy",
                f"{tag}_time.npy"
            ])

            if not success:
                print("Stopping — model ran out of memory.")
                break

    print("Finished. Results saved to:", csv_path)


if __name__ == "__main__":
    run_sweep()
