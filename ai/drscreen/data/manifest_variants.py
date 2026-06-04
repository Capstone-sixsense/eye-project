"""manifest CSV 변형(variant) 생성과 분할 통계 요약 유틸.

핵심은 'shadow validation' 만들기: 외부 테스트 도메인과 라벨 분포가 같은
검증셋을 학습 도메인에서 떼어내, 검증 성능이 외부 테스트를 더 잘 대변하도록 한다
(도메인 일반화 실험에서 검증-테스트 분포 차이를 줄이기 위함).

- build_shadow_validation_manifest: reference 분할(예: test)의 라벨별 개수만큼
  source 분할(예: train)에서 같은 도메인 행을 표본추출해 destination(예: val)으로 옮긴다.
- summarize_manifest_variant: 분할/도메인/라벨별 행 수를 집계한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ManifestVariantSummary:
    rows: int
    split_counts: dict[str, int]
    domain_split_counts: dict[str, dict[str, int]]
    domain_label_counts: dict[str, dict[int, int]]


def build_shadow_validation_manifest(
    manifest_path: str | Path,
    *,
    domain: str,
    source_split: str = "train",
    destination_split: str = "val",
    reference_split: str = "test",
    seed: int = 42,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required_columns = {"image_path", "label", "split", "domain"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    domain_reference = frame[(frame["domain"] == domain) & (frame["split"] == reference_split)]
    if domain_reference.empty:
        raise ValueError(f"No {domain} rows were found in reference split: {reference_split}")

    domain_source = frame[(frame["domain"] == domain) & (frame["split"] == source_split)]
    if domain_source.empty:
        raise ValueError(f"No {domain} rows were found in source split: {source_split}")

    updated = frame.copy()
    label_counts = {
        int(label): int(count)
        for label, count in domain_reference["label"].value_counts().sort_index().items()
    }
    sampled_indices: list[int] = []
    for offset, (label, count) in enumerate(label_counts.items()):
        candidates = domain_source[domain_source["label"] == label]
        if len(candidates) < count:
            raise ValueError(
                f"Not enough {domain} rows with label={label} in {source_split}: "
                f"required {count}, found {len(candidates)}"
            )
        sampled = candidates.sample(n=count, random_state=seed + offset)
        sampled_indices.extend(int(index) for index in sampled.index.tolist())

    updated.loc[sampled_indices, "split"] = destination_split
    updated = updated.sort_values(["split", "domain", "image_id"], kind="stable").reset_index(drop=True)
    return updated


def summarize_manifest_variant(frame: pd.DataFrame) -> ManifestVariantSummary:
    split_counts = {str(key): int(value) for key, value in frame["split"].value_counts().items()}
    domain_split_counts: dict[str, dict[str, int]] = {}
    for domain, subset in frame.groupby("domain"):
        domain_split_counts[str(domain)] = {
            str(key): int(value) for key, value in subset["split"].value_counts().items()
        }

    domain_label_counts: dict[str, dict[int, int]] = {}
    for domain, subset in frame.groupby("domain"):
        domain_label_counts[str(domain)] = {
            int(key): int(value) for key, value in subset["label"].value_counts().sort_index().items()
        }

    return ManifestVariantSummary(
        rows=len(frame),
        split_counts=split_counts,
        domain_split_counts=domain_split_counts,
        domain_label_counts=domain_label_counts,
    )


def write_shadow_validation_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    domain: str,
    source_split: str = "train",
    destination_split: str = "val",
    reference_split: str = "test",
    seed: int = 42,
) -> ManifestVariantSummary:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_shadow_validation_manifest(
        manifest_path,
        domain=domain,
        source_split=source_split,
        destination_split=destination_split,
        reference_split=reference_split,
        seed=seed,
    )
    frame.to_csv(output_path, index=False)
    return summarize_manifest_variant(frame)
