"""CBM(개념 병목) 모델의 개념 활성 다양성 진단.

각 개념(MA/HE/EX/SE) 채널이 서로 구별되는 패턴을 학습했는지(중복/붕괴되지 않았는지)를
엔트로피로 점검한다. CBM 연구 트랙의 '개념이 의미있게 분화됐는지' 게이트 용도다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def run(
    *,
    config_path: str,
    split: str = "train",
    mask_valid_only: bool = True,
    output: str | None = None,
) -> dict:
    from drscreen.infer.service import InferenceSession
    from drscreen.settings import get_run_evaluation_dir
    from drscreen.train.data_loader_factory import _build_datasets

    resolved_config = Path(config_path).resolve()
    project_root = resolved_config.parents[1]
    session = InferenceSession.from_config_path(resolved_config)
    config = session.config

    train_dataset, val_dataset, _manifest_path = _build_datasets(config, project_root)
    dataset = train_dataset if split == str(config["data"]["train_split"]) else val_dataset
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    channel_means: list[np.ndarray] = []
    entropies: list[float] = []
    kept = 0
    skipped = 0
    eps = 1e-8

    session.model.eval()
    with torch.no_grad():
        for batch in loader:
            if mask_valid_only:
                valid = batch.get("seg_mask_valid")
                if valid is None:
                    continue
                valid = valid.bool()
                if not valid.any():
                    skipped += int(valid.numel())
                    continue
                images = batch["image"][valid].to(session.device)
                skipped += int((~valid).sum().item())
            else:
                images = batch["image"].to(session.device)
            probs = session.model.predict_seg(images)
            means = probs.mean(dim=(2, 3)).detach().cpu().numpy()
            norm = means / np.clip(means.sum(axis=1, keepdims=True), eps, None)
            entropy = -(norm * np.log(np.clip(norm, eps, None))).sum(axis=1) / np.log(norm.shape[1])
            channel_means.extend(means)
            entropies.extend(entropy.astype(float).tolist())
            kept += int(means.shape[0])

    if kept == 0:
        raise ValueError("No samples available for CBM entropy diagnosis")

    channel_arr = np.stack(channel_means, axis=0)
    result = {
        "version": config.get("project", {}).get("version", "unknown"),
        "checkpoint_path": str(session.checkpoint_path),
        "split": split,
        "mask_valid_only": mask_valid_only,
        "n": kept,
        "skipped": skipped,
        "concept_codes": ["MA", "HE", "EX", "SE"],
        "channel_mean": channel_arr.mean(axis=0).astype(float).tolist(),
        "channel_std": channel_arr.std(axis=0).astype(float).tolist(),
        "normalized_entropy_mean": float(np.mean(entropies)),
        "normalized_entropy_std": float(np.std(entropies)),
        "redundant_solution_gate": {
            "threshold": 0.3,
            "pass": bool(float(np.mean(entropies)) >= 0.3),
        },
    }

    if output is None:
        eval_dir = get_run_evaluation_dir(project_root, str(result["version"]))
        eval_dir.mkdir(parents=True, exist_ok=True)
        output = str(eval_dir / f"cbm_entropy_{split}.json")
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CBM concept activation diversity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--all-rows", action="store_true", help="Do not filter to seg_mask_valid rows.")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        mask_valid_only=not args.all_rows,
        output=args.output,
    )


if __name__ == "__main__":
    main()
