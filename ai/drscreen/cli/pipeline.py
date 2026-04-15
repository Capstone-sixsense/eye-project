from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from drscreen.settings import ensure_runtime_directories, load_app_config
from drscreen.train.runner import describe_training_setup, run_split_evaluation, run_training
from drscreen.utils.logging import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ssl_pretrain → train → evaluate in one shot."
    )
    parser.add_argument("--ssl-config", default=None, help="Path to SSL pretraining YAML (optional).")
    parser.add_argument("--config", required=True, help="Path to fine-tuning YAML config.")
    parser.add_argument(
        "--split",
        default="external_test",
        help="Split to evaluate after training (default: external_test).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ft_config_path = Path(args.config).resolve()
    project_root = ft_config_path.parents[1]
    base_path = None
    candidate_base = ft_config_path.parent / "base.yaml"
    if ft_config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base

    if args.ssl_config:
        ssl_config_path = Path(args.ssl_config).resolve()
        ssl_config = load_app_config(ssl_config_path, base_path=None)

        if args.dry_run:
            print("SSL config loaded:", ssl_config_path)
        else:
            from drscreen.ssl.trainer import run_ssl_pretraining
            LOGGER.info("=== Step 1/3: SSL Pretraining ===")
            ssl_summary = run_ssl_pretraining(
                ssl_config, config_path=ssl_config_path, project_root=project_root
            )
            LOGGER.info("SSL complete. Encoder saved to: %s", ssl_summary["output_path"])
    else:
        LOGGER.info("--ssl-config not provided, skipping SSL pretraining.")

    ft_config = load_app_config(ft_config_path, base_path=base_path)
    ensure_runtime_directories(ft_config, project_root)

    step_prefix = "2/3" if args.ssl_config else "1/2"

    if args.dry_run:
        setup = describe_training_setup(ft_config, config_path=ft_config_path, project_root=project_root)
        pprint(setup)
        return

    LOGGER.info("=== Step %s: Fine-tuning ===", step_prefix)
    train_summary = run_training(ft_config, config_path=ft_config_path, project_root=project_root)
    LOGGER.info(
        "Training complete. best_epoch=%d best_val_auroc=%.4f",
        train_summary["best_epoch"],
        train_summary["best_val_auroc"],
    )

    eval_step = "3/3" if args.ssl_config else "2/2"
    LOGGER.info("=== Step %s: Evaluation on split=%s ===", eval_step, args.split)
    eval_summary = run_split_evaluation(
        ft_config,
        config_path=ft_config_path,
        project_root=project_root,
        split_name=args.split,
    )
    auroc = eval_summary["metrics"]["auroc"]
    sensitivity = eval_summary["metrics"]["sensitivity"]
    specificity = eval_summary["metrics"]["specificity"]
    LOGGER.info(
        "Evaluation complete. AUROC=%.4f sensitivity=%.4f specificity=%.4f",
        auroc, sensitivity, specificity,
    )
    print(f"\nResults on {args.split}:")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  Report      : {eval_summary.get('output_path', '')}")


if __name__ == "__main__":
    main()
