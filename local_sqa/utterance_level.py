import ast
from collections import defaultdict
import functools
import logging
import os
from pathlib import Path
import typing as tp

from joblib import Parallel, delayed
import numpy as np
import padertorch as pt
import psutil
import scipy
import torch
from tqdm import tqdm

from .modules.ssl_mos import SpeechQualityPredictor
from .modules.data_loader import JsonParser, Dataloader

LOCAL_SQA_INFER_MODEL = os.environ.get("LOCAL_SQA_INFER_MODEL", None)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def evaluate_frame_scores(frame_scores: np.ndarray, frame_lens, model):
    log_returns = np.diff(np.log(frame_scores))
    log_returns = pt.unpad_sequence(log_returns.T, frame_lens)
    sigma_returns = np.stack(list(map(np.std, log_returns)))
    volatility = sigma_returns * np.sqrt(frame_lens / model.model.encoder.frame_rate)
    return volatility


def worker(batch, model, device):
    batch = pt.data.example_to_device(batch, device)

    targets = torch.tensor(batch["mos"]).float().unsqueeze(1)

    preds, frame_scores, frame_lens = model(
        batch["audio"], batch["num_samples"],
    )
    if preds.ndim == 1:
        preds = preds[:, None]

    volatility = evaluate_frame_scores(frame_scores, frame_lens, model)
    return preds, targets, volatility


def main(
    database_path: str,
    model_dir: str,
    output_dir: tp.Optional[tp.Union[str, Path]] = None,
    checkpoint_name: str = "ckpt_best_SRCC.pth",
    num_workers: int = -1,
):
    if model_dir is None:
        if LOCAL_SQA_INFER_MODEL is None:
            raise RuntimeError(
                "Either provide --model_dir or set LOCAL_SQA_INFER_MODEL."
            )
        model_dir = LOCAL_SQA_INFER_MODEL

    logger.info(
        "Loading SQA model from %s/checkpoints/%s",
        model_dir,
        checkpoint_name,
    )
    model = SpeechQualityPredictor(
        storage_dir=model_dir,
        checkpoint_name=checkpoint_name,
        return_numpy=True,
        prepare_example_before=True,
    )

    if output_dir is None:
        output_dir = Path(model_dir) / "eval" / "utterance_level"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    json_path, dataset_names = database_path.split("::")
    try:
        dataset_names = ast.literal_eval(dataset_names)
    except ValueError:
        dataset_names = [dataset_names]

    parser = JsonParser(
        json_path,
        train_dataset_names=None,
        val_dataset_names=dataset_names,
        mos_key="rating.mean",
    )

    if num_workers < 0:
        num_workers = len(psutil.Process().cpu_affinity())

    test_loader = Dataloader(
        1,
        stage="val",
        shuffle=False,
        num_workers=num_workers,
        buffer_size=2 * num_workers,
    )(parser)

    all_targets = []
    all_preds = []
    volatilities = []

    iterator = (
        delayed(functools.partial(worker, model=model, device=device))(batch)
        for batch in tqdm(test_loader)
    )

    for preds, targets, volatility in Parallel(n_jobs=num_workers, backend="threading")(iterator):
        all_preds.append(preds)
        all_targets.append(targets)
        volatilities.append(volatility)

    all_targets = np.concatenate(all_targets).squeeze()
    all_preds = np.concatenate(all_preds).squeeze()
    volatilities = np.concatenate(volatilities)

    results = defaultdict(dict)
    results["volatility"] = float(volatilities.mean().item())
    logger.info("Frame scores volatility / second: %.3f", results["volatility"])

    MSE = float(np.mean((all_targets - all_preds) ** 2))
    LCC = float(np.corrcoef(all_targets, all_preds)[0][1])
    SRCC = float(scipy.stats.spearmanr(all_targets, all_preds).statistic)
    KTAU = float(scipy.stats.kendalltau(all_targets, all_preds).statistic)

    logger.info("[UTTERANCE] %sset error= %f", "+".join(dataset_names), MSE)
    logger.info("[UTTERANCE] Linear correlation coefficient= %f", LCC)
    logger.info("[UTTERANCE] Spearman rank correlation coefficient= %f", SRCC)
    logger.info("[UTTERANCE] Kendall Tau rank correlation coefficient= %f", KTAU)

    results["UTTERANCE"].update(
        {
            "mse": MSE,
            "lcc": LCC,
            "srcc": SRCC,
            "kendall_tau": KTAU,
        }
    )

    import paderbox as pb
    pb.io.dump_json(dict(results), output_dir / "results.json")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
