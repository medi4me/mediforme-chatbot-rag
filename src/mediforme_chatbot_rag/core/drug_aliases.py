"""한국어 약 식별자 → 영어 generic 매핑 (Phase 2 cross-lingual alias)

- 한국어 성분명·브랜드명 drug_id 로도 영어 FDA 라벨 청크에 닿도록 매핑
- 키는 소문자 비교, 값은 영어 generic (FDA generic_name 에 매칭되는 토큰)
- 인덱스에 아예 없는 약(예: 암브록솔)도 매핑은 두되, 매칭은 인덱스 보유 여부에 달림
"""

from __future__ import annotations

# 한국어 성분명 → 영어 generic
_INGREDIENT_KO_EN: dict[str, str] = {
    "아세트아미노펜": "acetaminophen",
    "이부프로펜": "ibuprofen",
    "아세틸살리실산": "aspirin",
    "펙소페나딘": "fexofenadine",
    "세티리진": "cetirizine",
    "로라타딘": "loratadine",
    "파모티딘": "famotidine",
    "메트포르민": "metformin",
    "암브록솔": "ambroxol",
    "세마글루티드": "semaglutide",
    "티르제파타이드": "tirzepatide",
    "니르마트렐비르": "nirmatrelvir",
    "레보노르게스트렐": "levonorgestrel",
    "레카네맙": "lecanemab",
    "아토르바스타틴": "atorvastatin",
    "로수바스타틴": "rosuvastatin",
    "암로디핀": "amlodipine",
    "에스오메프라졸": "esomeprazole",
    "세르트랄린": "sertraline",
    "시타글립틴": "sitagliptin",
    "레보티록신": "levothyroxine",
    "카르베딜롤": "carvedilol",
    "다비가트란": "dabigatran",
    "오셀타미비르": "oseltamivir",
    # 항암·biologics·특수약
    "아달리무맙": "adalimumab",
    "에타너셉트": "etanercept",
    "두필루맙": "dupilumab",
    "펨브롤리주맙": "pembrolizumab",
    "트라스투주맙": "trastuzumab",
    "타목시펜": "tamoxifen",
    "이매티닙": "imatinib",
    "암페타민": "amphetamine",
}

# 한국어 브랜드명 → 영어 generic
_BRAND_KO_EN: dict[str, str] = {
    "타이레놀": "acetaminophen",
    "부루펜": "ibuprofen",
    "알레그라": "fexofenadine",
    "지르텍": "cetirizine",
    "클라리틴": "loratadine",
    "가스터": "famotidine",
    "무코펙트": "ambroxol",
    "위고비": "semaglutide",
    "오젬픽": "semaglutide",
    "마운자로": "tirzepatide",
    "젭바운드": "tirzepatide",
    "팍스로비드": "nirmatrelvir",
    "플랜비": "levonorgestrel",
    "레켐비": "lecanemab",
    "노바스크": "amlodipine",
    "리피토": "atorvastatin",
    "크레스토": "rosuvastatin",
    "자누비아": "sitagliptin",
    "졸로프트": "sertraline",
    "넥시움": "esomeprazole",
    "다이아벡스": "metformin",
    "씬지로이드": "levothyroxine",
    "딜라트렌": "carvedilol",
    "프라닥사": "dabigatran",
    "타미플루": "oseltamivir",
    # 항암·biologics·특수약 브랜드
    "휴미라": "adalimumab",
    "엔브렐": "etanercept",
    "듀피젠트": "dupilumab",
    "키트루다": "pembrolizumab",
    "허셉틴": "trastuzumab",
    "글리벡": "imatinib",
    "애더럴": "amphetamine",
}

# 소문자 키로 통합
DRUG_ALIASES: dict[str, str] = {
    k.lower(): v for k, v in {**_INGREDIENT_KO_EN, **_BRAND_KO_EN}.items()
}


def expand_drug_id(drug_id: str) -> set[str]:
    """
    drug_id 를 매칭용 needle 집합으로 확장
    - 원본(소문자) + 알려진 영어 generic 매핑이 있으면 추가
    """
    base = drug_id.lower().strip()
    needles = {base}
    mapped = DRUG_ALIASES.get(base)
    if mapped:
        needles.add(mapped.lower())
    return needles
