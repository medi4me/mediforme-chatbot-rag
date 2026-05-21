"""일회성 백필: 기존 chunks.json 의 text 헤더에서 brand/generic alias 복원

- 기존 인덱스는 Chunk 모델에 brand_name/generic_name 필드가 생기기 전 빌드라
  메타가 비어 있으나, text 첫 줄 헤더 `[primary / alias1 / alias2]` 에 이름이 보존돼 있음
- 헤더를 파싱해 alias 를 generic_name/brand_name 필드로 채움 (필터는 세 필드 OR 매칭이라
  brand/generic 할당 순서는 무관, 이름 집합만 맞으면 정식 재빌드와 동작 동일)
- text 는 불변이므로 index.faiss 벡터는 그대로 재사용 (재임베딩 없음)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

INDEX_DIR = Path("data/index")


def parse_aliases(text: str, drug_name: str) -> tuple[str, str, bool]:
    first = text.splitlines()[0].strip()
    if not (first.startswith("[") and first.endswith("]")):
        return "", "", False  # 헤더 형식 아님
    inner = first[1:-1]
    if inner == drug_name:
        return "", "", True
    prefix = f"{drug_name} / "
    if inner.startswith(prefix):
        aliases = inner[len(prefix):].split(" / ")
        ok = True
    else:
        parts = inner.split(" / ")
        aliases = parts[1:]
        ok = bool(parts) and parts[0] == drug_name
    generic = aliases[0] if aliases else ""
    brand = " / ".join(aliases[1:]) if len(aliases) > 1 else ""
    return brand, generic, ok


def main() -> int:
    chunks_path = INDEX_DIR / "chunks.json"
    data = json.loads(chunks_path.read_text(encoding="utf-8"))

    total = len(data)
    with_alias = 0
    mismatches = 0
    for c in data:
        brand, generic, ok = parse_aliases(c["text"], c["drug_name"])
        c["brand_name"] = brand
        c["generic_name"] = generic
        if generic or brand:
            with_alias += 1
        if not ok:
            mismatches += 1

    if mismatches:
        print(f"경고: 헤더-drug_name 불일치 {mismatches}건", file=sys.stderr)

    chunks_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"백필 완료: {total} 청크 중 {with_alias} 청크에 alias 채움 (불일치 {mismatches})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
