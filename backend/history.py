"""
분석 이력 저장/조회 유틸리티 (SQLite + AES-256-GCM 암호화).

저장 구조:
  storage/history.db                        SQLite 데이터베이스 (메타데이터 + 로그)
  storage/raw_<id>.<ext>.enc                원본 이미지 (암호화, 디스크)
  results/<YYYY-MM-DD>/report_<id>.png.enc  분석 결과 이미지 (암호화, 디스크)

<id>는 'YYYYMMDD_HHMMSS_mmm' 형식의 타임스탬프 문자열 (KST 기준).
<YYYY-MM-DD> 폴더도 KST 날짜 기준.

DB 스키마 (records 테이블):
  id                TEXT PRIMARY KEY   -- 시간 정렬 가능 문자열 (KST 기반)
  created_at        TEXT               -- ISO 8601 KST (UTC+9)
  original_filename TEXT
  raw_ext           TEXT               -- 원본 이미지 확장자
  raw_path          TEXT               -- 디스크 경로 (NULL = 파일 없음)
  report_path       TEXT               -- 디스크 경로 (NULL = 파일 없음)
  metadata_enc      BLOB               -- AES-256-GCM 암호화된 JSON

DB 스키마 (logs 테이블):
  id       INTEGER PRIMARY KEY AUTOINCREMENT
  ts       TEXT    -- ISO 8601 KST (UTC+9)
  level    TEXT    -- INFO / WARNING / ERROR
  phase    TEXT    -- startup / upload / fundus_check / quickqual / inference / report / done
  job_id   TEXT    -- 분석 job_id (없으면 NULL)
  message  TEXT
  elapsed  REAL    -- 소요 시간(초), 해당되는 경우만
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import crypto

UPLOAD_DIR = "storage"
RESULTS_DIR = "results"
DB_PATH = os.path.join(UPLOAD_DIR, "history.db")

# 기존 .json.enc 파일 인식용 패턴 (migration.py에서 참조)
ID_PATTERN = re.compile(r"^report_(?P<id>\d{8}_\d{6}_\d{3})\.json\.enc$")
RECORD_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{3}$")

_KST = timezone(timedelta(hours=9))
_db: sqlite3.Connection | None = None
_write_lock = threading.Lock()


# ---------------------------------------------------------------
# DB 초기화 / 종료
# ---------------------------------------------------------------
def init_db() -> None:
    """DB 연결을 열고 테이블·인덱스를 생성한다. lifespan에서 1회 호출."""
    global _db
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _db = sqlite3.connect(DB_PATH, check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("PRAGMA synchronous=NORMAL")
    _db.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id                TEXT PRIMARY KEY,
            created_at        TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            raw_ext           TEXT NOT NULL DEFAULT 'png',
            raw_path          TEXT,
            report_path       TEXT,
            metadata_enc      BLOB NOT NULL
        )
    """)
    _db.execute("CREATE INDEX IF NOT EXISTS idx_id_desc ON records(id DESC)")
    _db.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT    NOT NULL,
            level    TEXT    NOT NULL DEFAULT 'INFO',
            phase    TEXT,
            job_id   TEXT,
            message  TEXT    NOT NULL,
            elapsed  REAL
        )
    """)
    _db.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts     ON logs(ts DESC)")
    _db.execute("CREATE INDEX IF NOT EXISTS idx_logs_job_id ON logs(job_id)")
    cutoff = (datetime.now(_KST) - timedelta(days=30)).isoformat(timespec="milliseconds")
    _db.execute("DELETE FROM logs WHERE ts < ?", (cutoff,))
    _db.commit()


def close_db() -> None:
    """DB 연결 종료. lifespan 종료 시 호출."""
    global _db
    if _db is not None:
        _db.close()
        _db = None


# ---------------------------------------------------------------
# 로그
# ---------------------------------------------------------------
def write_log(
    message: str,
    *,
    level: str = "INFO",
    phase: str | None = None,
    job_id: str | None = None,
    elapsed: float | None = None,
) -> None:
    if _db is None:
        return
    ts = datetime.now(_KST).isoformat(timespec="milliseconds")
    with _write_lock:
        _db.execute(
            "INSERT INTO logs (ts, level, phase, job_id, message, elapsed) VALUES (?,?,?,?,?,?)",
            (ts, level, phase, job_id, message, elapsed),
        )
        _db.commit()


def list_logs(
    limit: int = 100,
    offset: int = 0,
    level: str | None = None,
    job_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if _db is None:
        return [], 0
    conditions: list[str] = []
    params: list[Any] = []
    if level:
        conditions.append("level = ?")
        params.append(level)
    if job_id:
        conditions.append("job_id = ?")
        params.append(job_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    total: int = _db.execute(f"SELECT COUNT(*) FROM logs {where}", params).fetchone()[0]
    rows = _db.execute(
        f"SELECT id, ts, level, phase, job_id, message, elapsed FROM logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    items = [
        {"id": r[0], "ts": r[1], "level": r[2], "phase": r[3],
         "job_id": r[4], "message": r[5], "elapsed": r[6]}
        for r in rows
    ]
    return items, total


# ---------------------------------------------------------------
# ID 발급
# ---------------------------------------------------------------
def make_record_id(now: datetime | None = None) -> str:
    """타임스탬프 기반 분석 ID 생성. 예: '20260428_165403_123'."""
    now = now or datetime.now(_KST)
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def _date_subdir(record_id: str) -> str:
    """record_id에서 'YYYY-MM-DD' 폴더명 추출."""
    return f"{record_id[0:4]}-{record_id[4:6]}-{record_id[6:8]}"


def _aad(record_id: str) -> bytes:
    """GCM associated data — record_id를 묶어 파일 swap 공격 방어."""
    return record_id.encode("utf-8")


# ---------------------------------------------------------------
# 경로 헬퍼
# ---------------------------------------------------------------
def raw_path_for(record_id: str, ext: str = "png") -> str:
    return os.path.join(UPLOAD_DIR, f"raw_{record_id}.{ext}{crypto.ENC_SUFFIX}")


def report_image_path_for(record_id: str) -> str:
    return os.path.join(
        RESULTS_DIR,
        _date_subdir(record_id),
        f"report_{record_id}.png{crypto.ENC_SUFFIX}",
    )


def metadata_path_for(record_id: str) -> str:
    """레거시 .json.enc 경로 (migration.py 참조용)."""
    return os.path.join(
        RESULTS_DIR,
        _date_subdir(record_id),
        f"report_{record_id}.json{crypto.ENC_SUFFIX}",
    )


# ---------------------------------------------------------------
# 저장
# ---------------------------------------------------------------
def save_raw_image(record_id: str, data: bytes, ext: str = "png") -> str:
    """원본 이미지 바이트를 암호화 저장. 저장 경로 반환."""
    path = raw_path_for(record_id, ext=ext)
    return crypto.write_encrypted_file(path, data, associated_data=_aad(record_id))


def save_report_image(record_id: str, data: bytes) -> str:
    """리포트 이미지(PNG 바이트)를 암호화 저장. 저장 경로 반환."""
    path = report_image_path_for(record_id)
    return crypto.write_encrypted_file(path, data, associated_data=_aad(record_id))


def save_metadata(
    record_id: str,
    *,
    original_filename: str,
    raw_url: str,
    report_url: str,
    label: str | None,
    abnormal_probability: float,
    quality: dict[str, Any] | None,
    metrics: dict,
    evidence: dict | None = None,
    raw_ext: str = "png",
) -> None:
    """분석 메타데이터를 SQLite에 저장."""
    if _db is None:
        raise RuntimeError("DB가 초기화되지 않았습니다.")
    created_at = datetime.now(_KST).isoformat(timespec="seconds")
    payload = {
        "id": record_id,
        "created_at": created_at,
        "original_filename": original_filename,
        "raw_url": raw_url,
        "report_url": report_url,
        "label": label,
        "abnormal_probability": abnormal_probability,
        "quality": quality,
        "metrics": metrics,
        "evidence": evidence,
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    metadata_enc = crypto.encrypt_bytes(blob, associated_data=_aad(record_id))

    with _write_lock:
        _db.execute(
            """
            INSERT OR REPLACE INTO records
                (id, created_at, original_filename, raw_ext, raw_path, report_path, metadata_enc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                created_at,
                original_filename,
                raw_ext,
                raw_path_for(record_id, ext=raw_ext),
                report_image_path_for(record_id),
                metadata_enc,
            ),
        )
        _db.commit()


# ---------------------------------------------------------------
# 조회
# ---------------------------------------------------------------
def load_record(record_id: str) -> dict[str, Any] | None:
    """단일 이력 조회. 메타데이터 복호화 + 파일 존재 여부 검증."""
    if _db is None:
        return None
    row = _db.execute(
        "SELECT raw_path, report_path, metadata_enc FROM records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if row is None:
        return None

    raw_path, report_path, metadata_enc = row

    try:
        plaintext = crypto.decrypt_bytes(
            bytes(metadata_enc), associated_data=_aad(record_id)
        )
        payload = json.loads(plaintext.decode("utf-8"))
    except (crypto.DecryptionFailed, json.JSONDecodeError):
        return None

    # 파일 존재 검증 (DB에 기록된 경로 기준)
    payload["raw_url"] = (
        f"/image/raw/{record_id}"
        if raw_path and os.path.exists(raw_path)
        else None
    )
    payload["report_url"] = (
        f"/image/report/{record_id}"
        if report_path and os.path.exists(report_path)
        else None
    )
    return payload


def list_records(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """저장된 분석 이력을 최신순으로 반환. 페이지네이션 지원."""
    records, _ = list_records_with_total(limit=limit, offset=offset)
    return records


def list_records_with_total(
    limit: int = 50, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """저장된 분석 이력을 최신순으로 반환하고 전체 건수를 함께 돌려준다."""
    if _db is None:
        return [], 0
    total: int = _db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    rows = _db.execute(
        "SELECT id, raw_path, report_path, metadata_enc FROM records ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()

    records: list[dict[str, Any]] = []
    for rid, raw_path, report_path, metadata_enc in rows:
        try:
            plaintext = crypto.decrypt_bytes(
                bytes(metadata_enc), associated_data=_aad(rid)
            )
            payload = json.loads(plaintext.decode("utf-8"))
        except (crypto.DecryptionFailed, json.JSONDecodeError):
            continue
        payload["raw_url"] = (
            f"/image/raw/{rid}"
            if raw_path and os.path.exists(raw_path)
            else None
        )
        payload["report_url"] = (
            f"/image/report/{rid}"
            if report_path and os.path.exists(report_path)
            else None
        )
        records.append(payload)
    return records, total


def load_raw_bytes(record_id: str) -> tuple[bytes, str] | None:
    """raw 이미지 바이트와 원본 확장자를 반환. 없으면 None."""
    if _db is None:
        return None
    row = _db.execute(
        "SELECT raw_path, raw_ext FROM records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None

    raw_path, raw_ext = row
    if not os.path.exists(raw_path):
        return None
    try:
        data = crypto.read_encrypted_file(raw_path, associated_data=_aad(record_id))
        return data, raw_ext
    except (OSError, crypto.DecryptionFailed):
        return None


def load_report_bytes(record_id: str) -> bytes | None:
    if _db is None:
        return None
    row = _db.execute(
        "SELECT report_path FROM records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None

    report_path = row[0]
    if not os.path.exists(report_path):
        return None
    try:
        return crypto.read_encrypted_file(report_path, associated_data=_aad(record_id))
    except (OSError, crypto.DecryptionFailed):
        return None


def count_records() -> int:
    if _db is None:
        return 0
    return _db.execute("SELECT COUNT(*) FROM records").fetchone()[0]


def insert_migrated_record(
    record_id: str,
    created_at: str,
    original_filename: str,
    raw_ext: str,
    raw_path: str | None,
    report_path: str | None,
    metadata_enc_bytes: bytes,
) -> bool:
    """migration.py 전용 헬퍼 — INSERT OR IGNORE. 삽입 시 True, 스킵 시 False."""
    if _db is None:
        return False
    with _write_lock:
        cur = _db.execute(
            """
            INSERT OR IGNORE INTO records
                (id, created_at, original_filename, raw_ext,
                 raw_path, report_path, metadata_enc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (record_id, created_at, original_filename, raw_ext,
             raw_path, report_path, metadata_enc_bytes),
        )
        _db.commit()
    return cur.rowcount > 0


def delete_record_files(record_id: str) -> None:
    """DB 레코드 제거 후 raw + report 파일 삭제."""
    if _db is None:
        return
    row = _db.execute(
        "SELECT raw_path, report_path FROM records WHERE id = ?",
        (record_id,),
    ).fetchone()

    with _write_lock:
        _db.execute("DELETE FROM records WHERE id = ?", (record_id,))
        _db.commit()

    if row:
        for path in row:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
