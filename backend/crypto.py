"""
AES-256-GCM 기반 파일 암복호화 유틸리티.

설계:
- 알고리즘: AES-256-GCM (AEAD, NIST 표준)
- 키:      환경변수 IMAGE_ENCRYPTION_KEY (base64-encoded 32 bytes)
- 파일 포맷: [12바이트 nonce][16바이트 GCM tag][암호문]
            (cryptography의 AESGCM은 ciphertext와 tag를 합쳐서 돌려주므로
             실제 디스크 포맷은 [nonce(12) | ciphertext+tag])

보안 주의:
- nonce는 매 암호화마다 새로 생성한다 (os.urandom).
  같은 키 + 같은 nonce를 두 번 쓰면 GCM의 보안이 완전히 깨진다.
- 인증 태그가 자동으로 검증되므로, 위변조된 파일은 복호화 시 예외가 난다.
"""
from __future__ import annotations

import base64
import os
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 파일 확장자 (암호화된 파일에 일관되게 붙임)
ENC_SUFFIX: Final = ".enc"

# GCM 표준 권장 nonce 길이 (RFC 5116)
NONCE_BYTES: Final = 12

_ENV_KEY = "IMAGE_ENCRYPTION_KEY"


class CryptoNotConfigured(RuntimeError):
    """환경변수 IMAGE_ENCRYPTION_KEY 가 없거나 형식이 잘못됨."""


class DecryptionFailed(RuntimeError):
    """복호화 실패 — 키 불일치, 파일 위변조, 또는 손상."""


def _load_key() -> bytes:
    """환경변수에서 32바이트 키를 base64 디코딩해서 반환."""
    raw = os.environ.get(_ENV_KEY, "").strip()
    if not raw:
        raise CryptoNotConfigured(
            f"환경변수 {_ENV_KEY}가 비어있습니다. "
            f".env 파일에 base64 인코딩된 32바이트 키를 설정하세요."
        )
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise CryptoNotConfigured(
            f"{_ENV_KEY} 디코딩 실패 (base64 형식 오류): {exc}"
        ) from exc
    if len(key) != 32:
        raise CryptoNotConfigured(
            f"{_ENV_KEY} 디코딩 결과가 {len(key)}바이트입니다. "
            "AES-256-GCM은 정확히 32바이트 키가 필요합니다."
        )
    return key


# 모듈 로드 시 1회 검증 (서버 시작 시점에 미리 실패하도록 유도)
_KEY = _load_key()
_AESGCM = AESGCM(_KEY)


# ---------------------------------------------------------------
# 핵심 API: 바이트 단위 암복호화
# ---------------------------------------------------------------
def encrypt_bytes(plaintext: bytes, *, associated_data: bytes | None = None) -> bytes:
    """
    평문 바이트를 암호화. 결과 포맷: [nonce(12) | ciphertext+tag].

    Args:
        plaintext:        암호화할 데이터.
        associated_data:  선택. 무결성 보호 대상(평문에는 포함 안 됨)에 함께 묶고 싶은 데이터.
                          예: record_id를 묶어두면 다른 record의 파일을 갖다 붙여도 복호화 실패.
    """
    nonce = os.urandom(NONCE_BYTES)
    ct = _AESGCM.encrypt(nonce, plaintext, associated_data)
    return nonce + ct


def decrypt_bytes(blob: bytes, *, associated_data: bytes | None = None) -> bytes:
    """위 포맷의 암호문을 복호화. 위변조나 키 불일치 시 DecryptionFailed."""
    if len(blob) < NONCE_BYTES + 16:
        raise DecryptionFailed("암호문이 너무 짧습니다 (헤더 누락).")
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    try:
        return _AESGCM.decrypt(nonce, ct, associated_data)
    except Exception as exc:
        raise DecryptionFailed(f"복호화 실패: {type(exc).__name__}") from exc


# ---------------------------------------------------------------
# 파일 단위 헬퍼
# ---------------------------------------------------------------
def encrypted_path(plain_path: str) -> str:
    """평문 경로 → 암호화 파일 경로 (.enc 추가)."""
    return plain_path if plain_path.endswith(ENC_SUFFIX) else plain_path + ENC_SUFFIX


def write_encrypted_file(
    plain_path: str,
    data: bytes,
    *,
    associated_data: bytes | None = None,
) -> str:
    """
    data를 암호화해서 plain_path + '.enc' 로 저장하고, 저장 경로 반환.
    상위 디렉토리는 자동 생성.
    """
    enc_path = encrypted_path(plain_path)
    os.makedirs(os.path.dirname(enc_path) or ".", exist_ok=True)

    encrypted = encrypt_bytes(data, associated_data=associated_data)

    # 원자적 쓰기: 임시 파일에 먼저 쓰고 rename
    # (쓰는 도중 컨테이너가 죽어도 부분 파일이 남지 않음)
    tmp_path = enc_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(encrypted)
    os.replace(tmp_path, enc_path)
    return enc_path


def read_encrypted_file(
    enc_path: str,
    *,
    associated_data: bytes | None = None,
) -> bytes:
    """암호화된 파일을 읽어 복호화한 평문 바이트 반환."""
    with open(enc_path, "rb") as f:
        blob = f.read()
    return decrypt_bytes(blob, associated_data=associated_data)