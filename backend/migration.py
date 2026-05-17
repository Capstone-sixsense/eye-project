"""
기존 results/**/*.json.enc 파일을 SQLite(history.db)로 이전하는 1회성 스크립트.

실행 방법:
  cd /home/.../eye-project/backend
  IMAGE_ENCRYPTION_KEY=<your_key> python migration.py

동작:
  1. results/ 아래의 모든 .json.enc 파일을 탐색
  2. 각 파일을 복호화해 plaintext 컬럼(created_at, original_filename 등) 추출
  3. 암호화 원본 바이트를 metadata_enc BLOB으로 그대로 저장 (재암호화 없음)
  4. raw_path / report_path 파일 존재 여부 확인 후 경로 기록
  5. 이미 DB에 있는 레코드는 스킵 (INSERT OR IGNORE)
"""
from __future__ import annotations

import json
import os
import sys

# 실행 위치가 backend/ 가 아닐 경우를 대비해 경로 보정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crypto
import history


def migrate() -> None:
    history.init_db()

    if not os.path.isdir(history.RESULTS_DIR):
        print("results/ 디렉터리가 없습니다. 이전할 데이터가 없습니다.")
        return

    migrated = skipped = errors = 0

    with os.scandir(history.RESULTS_DIR) as top:
        for date_entry in top:
            if not date_entry.is_dir():
                continue
            for fname in os.listdir(date_entry.path):
                m = history.ID_PATTERN.match(fname)
                if not m:
                    continue

                record_id = m.group("id")
                meta_path = os.path.join(date_entry.path, fname)

                # 암호화 원본 바이트 읽기 (재암호화 없이 BLOB에 그대로 저장)
                try:
                    with open(meta_path, "rb") as f:
                        metadata_enc_bytes = f.read()
                    plaintext = crypto.decrypt_bytes(
                        metadata_enc_bytes,
                        associated_data=record_id.encode("utf-8"),
                    )
                    payload = json.loads(plaintext.decode("utf-8"))
                except Exception as exc:
                    print(f"[ERROR] {record_id}: {exc}")
                    errors += 1
                    continue

                created_at = payload.get("created_at", "")
                original_filename = payload.get("original_filename", "")

                # raw 파일 탐색 (확장자 불명이므로 파일명 패턴으로 탐색)
                raw_path: str | None = None
                raw_ext = "png"
                if os.path.isdir(history.UPLOAD_DIR):
                    for rf in os.listdir(history.UPLOAD_DIR):
                        if (
                            rf.startswith(f"raw_{record_id}.")
                            and rf.endswith(crypto.ENC_SUFFIX)
                        ):
                            raw_path = os.path.join(history.UPLOAD_DIR, rf)
                            stem = rf[: -len(crypto.ENC_SUFFIX)]
                            raw_ext = stem.rsplit(".", 1)[-1] if "." in stem else "png"
                            break

                # report 파일 경로
                rp = history.report_image_path_for(record_id)
                report_path: str | None = rp if os.path.exists(rp) else None

                inserted = history.insert_migrated_record(
                    record_id,
                    created_at,
                    original_filename,
                    raw_ext,
                    raw_path,
                    report_path,
                    metadata_enc_bytes,
                )
                if inserted:
                    migrated += 1
                    print(f"[OK] {record_id}")
                else:
                    skipped += 1

    print(f"\n완료 — 이전: {migrated}건 / 스킵(기존재): {skipped}건 / 오류: {errors}건")
    history.close_db()


if __name__ == "__main__":
    migrate()
