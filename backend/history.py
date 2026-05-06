"""
분석 이력 저장/조회 유틸리티 (AES-256-GCM 암호화).
파일 시스템 기반의 단순 저장소.

파일 배치 규약 (모든 산출물이 .enc 로 끝남):
  storage/raw_<id>.<ext>.enc                     원본 이미지(암호화)
  results/<YYYY-MM-DD>/report_<id>.png.enc       분석 결과 이미지(암호화)
  results/<YYYY-MM-DD>/report_<id>.json.enc      메타데이터(암호화)

<id>는 'YYYYMMDD_HHMMSS_mmm' 형식의 타임스탬프 문자열.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import crypto

UPLOAD_DIR = "storage"
RESULTS_DIR = "results"

# 메타데이터 암호 파일은 'report_<id>.json.enc'
ID_PATTERN = re.compile(r"^report_(?P<id>\d{8}_\d{6}_\d{3})\.json\.enc$")

# ---------------------------------------------------------------
# ID 발급
# ---------------------------------------------------------------
def make_record_id(now: datetime | None = None) -> str:
    """타임스탬프 기반 분석 ID 생성. 예: '20260428_165403_123'."""
    now = now or datetime.now()
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

# save_metadata
# ---------------------------------------------------------------
# 단건 분석의 메타데이터(라벨/확률/품질/지표 등)를 JSON 파일로 저장한다.
#
# [역할]
#   - 결과 이미지(.png)와 짝을 이루는 .json 파일을 만들어, 이력 조회 시
#     이미지뿐 아니라 수치 데이터까지 함께 돌려줄 수 있게 한다.
#   - 모든 키워드 인자를 강제(*)해 호출부의 가독성과 안전성을 높인다.
#
# [동작 순서]
#   1) 호출 시점의 timestamp(created_at)와 인자들을 합쳐 payload dict 구성.
#   2) metadata_path_for(record_id)로 저장 경로 결정
#      → results/<YYYY-MM-DD>/report_<id>.json
#   3) 상위 폴더가 없으면 생성 (exist_ok=True 라 충돌 없음).
#   4) UTF-8 + ensure_ascii=False 로 한글이 그대로 보이게 직렬화하여 기록.
#   5) 저장된 절대/상대 경로 문자열 반환.
#
# [반환값]
#   저장된 JSON 파일 경로 (디버깅/로깅 용도. main.py는 보통 이 값을 사용하지 않음)
#
# [참고]
#   - record_id가 같은 호출이 두 번 들어오면 마지막 호출이 덮어쓴다(last-write-wins).
def save_metadata(
    record_id: str,
    *,
    original_filename: str,
    raw_url: str,
    report_url: str,
    label: str | None,
    abnormal_probability: float,
    quality: dict | None,
    metrics: dict,
) -> str:
    """분석 메타데이터를 JSON으로 저장. 저장 경로 반환."""
    payload = {
        "id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_filename": original_filename,
        "raw_url": raw_url,
        "report_url": report_url,
        "label": label,
        "abnormal_probability": abnormal_probability,
        "quality": quality,
        "metrics": metrics,
    }
    path = metadata_path_for(record_id)
    blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return crypto.write_encrypted_file(path, blob, associated_data=_aad(record_id))


# ---------------------------------------------------------------
# 조회
# ---------------------------------------------------------------
# _iter_metadata_files (내부 헬퍼)
# ---------------------------------------------------------------
# results/ 아래 모든 날짜 폴더를 훑어 메타데이터 JSON 목록을 모은다.
#
# [역할]
#   - list_records / count_records 가 공통으로 쓰는 디스크 스캔 로직.
#
# [동작 순서]
#   1) results/ 디렉토리가 아예 없으면 빈 리스트 즉시 반환 (예: 첫 부팅 시).
#   2) results/ 바로 아래의 항목을 순회한다.
#      - 디렉토리가 아닌 항목(.gitkeep 등)은 isdir 체크로 스킵.
#   3) 각 날짜 폴더 안의 파일명을 ID_PATTERN(report_<id>.json)으로 매칭.
#   4) 매칭된 것만 (record_id, 메타데이터_경로) 튜플로 모아 반환.
#
# [반환값]
#   [(record_id, metadata_path), ...] 형태의 리스트 (정렬되지 않은 상태).
#
# [주의]
#   - 매 호출마다 디스크 I/O가 발생한다. 이력이 매우 많을 때는 캐싱 고려.
#   - 같은 record_id가 두 번 등장할 일은 없다(파일 시스템 특성상 유일).
def _iter_metadata_files() -> list[tuple[str, str]]:
    """
    results/ 아래의 모든 날짜 폴더를 순회하여
    (record_id, metadata_path) 튜플 리스트를 반환.
    """
    out: list[tuple[str, str]] = []
    if not os.path.isdir(RESULTS_DIR):
        return out

    for date_folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, date_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            m = ID_PATTERN.match(fname)
            if m:
                out.append((m.group("id"), os.path.join(folder_path, fname)))
    return out

# ---------------------------------------------------------------
# list_records
# ---------------------------------------------------------------
# 저장된 분석 이력을 최신순으로 페이지네이션해서 돌려준다.
#
# [역할]
#   - 프론트의 "이력 조회" 화면에 띄울 목록을 만들어주는 메인 함수.
#   - GET /history 엔드포인트의 핵심 로직.
#
# [동작 순서]
#   1) _iter_metadata_files()로 (record_id, path) 쌍 전체를 모은다.
#   2) record_id 문자열 기준 내림차순 정렬.
#      - record_id가 'YYYYMMDD_HHMMSS_mmm' 포맷(zero-padded)이라
#        문자열 정렬이 곧 시간 정렬이 된다 (사전식 정렬 == 시간 정렬).
#   3) offset 부터 limit 개수만큼 슬라이싱.
#   4) 각 record_id에 대해 load_record로 메타데이터 + 파일 존재성을 검증한 dict를 얻고,
#      None이 아닌 것만 결과에 포함.
#      - 메타 JSON이 손상됐거나 사라진 경우는 자동으로 누락 처리.
#
# [인자]
#   limit  : 한 페이지에 반환할 최대 개수 (main.py에서 1..200 범위 검증)
#   offset : 건너뛸 개수 (페이지 N → offset = N * limit)
#
# [반환값]
#   메타데이터 dict들의 리스트. 각 dict의 스키마는 save_metadata에서 저장한 그것과 동일.
def list_records(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """저장된 분석 이력을 최신순으로 반환. 페이지네이션 지원."""
    pairs = _iter_metadata_files()
    # ID가 타임스탬프 문자열이라 단순 역순 정렬 = 최신순
    pairs.sort(key=lambda p: p[0], reverse=True)

    sliced = pairs[offset : offset + limit]
    records: list[dict[str, Any]] = []
    for rid, _ in sliced:
        record = load_record(rid)
        if record is not None:
            records.append(record)
    return records

# ---------------------------------------------------------------
# load_record
# ---------------------------------------------------------------
# 단건 이력을 디스크에서 읽어 dict로 돌려준다. 파일 존재성도 함께 검증.
#
# [역할]
#   - GET /history/{id} 의 핵심 로직.
#   - list_records 도 내부적으로 이 함수를 사용해 항목 단위로 데이터를 만든다.
#
# [동작 순서]
#   1) metadata_path_for(record_id) 로 메타 JSON 경로를 계산.
#   2) 해당 파일이 없으면 None 반환 (호출부는 404로 변환).
#   3) JSON을 읽어 payload dict 로 파싱. 파일 손상/IO 에러 시 None 반환.
#   4) raw_url 검증:
#      - 메타에 적힌 경로가 실제로 존재하면 그대로 둔다.
#      - 사라졌다면 storage/ 안에서 'raw_<record_id>.*' 패턴으로 재탐색.
#        (확장자가 jpg→png 등으로 바뀌었을 가능성에 대한 fallback)
#      - 끝까지 못 찾으면 raw_url 을 None 으로 만들어 응답에 그대로 반환.
#        프론트는 None 처리(이미지 자리 비움 또는 placeholder)를 해야 한다.
#   5) report_url 도 동일한 검증을 거친다(단순 존재 여부만).
#   6) 검증된 payload 반환.
#
# [반환값]
#   dict | None
#   - dict: save_metadata로 저장한 그 형태에서 raw_url/report_url만 보정됨.
#   - None: 메타 자체가 없거나 손상된 경우.
def load_record(record_id: str) -> dict[str, Any] | None:
    """단일 이력 조회. 메타데이터 + 파일 존재 여부 검증."""
    meta_path = metadata_path_for(record_id)
    if not os.path.exists(meta_path):
        return None

    try:
        plaintext = crypto.read_encrypted_file(
            meta_path, associated_data=_aad(record_id)
        )
        payload = json.loads(plaintext.decode("utf-8"))
    except (OSError, json.JSONDecodeError, crypto.DecryptionFailed):
        return None
    
    return payload

def load_raw_bytes(record_id: str) -> tuple[bytes, str] | None:
    """
    raw 이미지 바이트와 원본 확장자를 반환. 없으면 None.
    """
    if not os.path.isdir(UPLOAD_DIR):
        return None
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(f"raw_{record_id}.") and fname.endswith(crypto.ENC_SUFFIX):
            path = os.path.join(UPLOAD_DIR, fname)
            try:
                data = crypto.read_encrypted_file(
                    path, associated_data=_aad(record_id)
                )
            except (OSError, crypto.DecryptionFailed):
                return None
            # raw_<id>.<ext>.enc 에서 <ext> 추출
            stem = fname[: -len(crypto.ENC_SUFFIX)]      # raw_<id>.<ext>
            ext = stem.rsplit(".", 1)[-1] if "." in stem else "png"
            return data, ext
    return None    

def load_report_bytes(record_id: str) -> bytes | None:
    path = report_image_path_for(record_id)
    if not os.path.exists(path):
        return None
    try:
        return crypto.read_encrypted_file(path, associated_data=_aad(record_id))
    except (OSError, crypto.DecryptionFailed):
        return None

def count_records() -> int:
    return len(_iter_metadata_files())


def delete_record_files(record_id: str) -> None:
    """raw + report + metadata 파일 모두 삭제 (없는 건 스킵)."""
    # raw
    if os.path.isdir(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(f"raw_{record_id}."):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, fname))
                except OSError:
                    pass
    for path in (report_image_path_for(record_id), metadata_path_for(record_id)):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass