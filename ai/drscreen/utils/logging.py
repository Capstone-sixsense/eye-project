"""공용 로거 헬퍼: 모듈 이름으로 표준 포맷 로거를 반환한다."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
