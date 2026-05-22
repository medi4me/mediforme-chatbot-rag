"""한국어 약 식별자 → 영어 generic 매핑 (cross-lingual alias)

- 한국어 성분명·브랜드명 drug_id 로도 영어 FDA 라벨 청크에 닿도록 매핑
- 키는 소문자 비교, 값은 영어 generic (FDA generic_name 에 매칭되는 토큰)
- 인덱스에 아예 없는 약(시드 갭)도 매핑은 두되, 매칭은 인덱스 보유 여부에 달림
"""

from __future__ import annotations

# 한국어 성분명 → 영어 generic
_INGREDIENT_KO_EN: dict[str, str] = {
    # 진통·소염·해열
    "아세트아미노펜": "acetaminophen",
    "이부프로펜": "ibuprofen",
    "아세틸살리실산": "aspirin",
    "나프록센": "naproxen",
    "세레콕시브": "celecoxib",
    "트라마돌": "tramadol",
    # 알레르기
    "펙소페나딘": "fexofenadine",
    "세티리진": "cetirizine",
    "로라타딘": "loratadine",
    # 소화기 (PPI/H2)
    "파모티딘": "famotidine",
    "에스오메프라졸": "esomeprazole",
    "오메프라졸": "omeprazole",
    "란소프라졸": "lansoprazole",
    "판토프라졸": "pantoprazole",
    "암브록솔": "ambroxol",
    # 순환·대사
    "암로디핀": "amlodipine",
    "로사르탄": "losartan",
    "카르베딜롤": "carvedilol",
    "아토르바스타틴": "atorvastatin",
    "로수바스타틴": "rosuvastatin",
    "푸로세미드": "furosemide",
    "사쿠비트릴": "sacubitril",
    # 당뇨
    "메트포르민": "metformin",
    "글리메피리드": "glimepiride",
    "시타글립틴": "sitagliptin",
    "리나글립틴": "linagliptin",
    "엠파글리플로진": "empagliflozin",
    "다파글리플로진": "dapagliflozin",
    "세마글루티드": "semaglutide",
    "티르제파타이드": "tirzepatide",
    "리라글루티드": "liraglutide",
    "둘라글루티드": "dulaglutide",
    # 갑상선·정신·수면
    "레보티록신": "levothyroxine",
    "세르트랄린": "sertraline",
    "에스시탈로프람": "escitalopram",
    "알프라졸람": "alprazolam",
    "졸피뎀": "zolpidem",
    "메틸페니데이트": "methylphenidate",
    "암페타민": "amphetamine",
    # 항응고
    "다비가트란": "dabigatran",
    "아픽사반": "apixaban",
    "리바록사반": "rivaroxaban",
    # 항바이러스
    "오셀타미비르": "oseltamivir",
    "니르마트렐비르": "nirmatrelvir",
    "몰누피라비르": "molnupiravir",
    "렘데시비르": "remdesivir",
    "빅테그라비르": "bictegravir",
    # 신경
    "레카네맙": "lecanemab",
    # 항암
    "펨브롤리주맙": "pembrolizumab",
    "니볼루맙": "nivolumab",
    "이필리무맙": "ipilimumab",
    "트라스투주맙": "trastuzumab",
    "퍼투주맙": "pertuzumab",
    "베바시주맙": "bevacizumab",
    "리툭시맙": "rituximab",
    "세툭시맙": "cetuximab",
    "사시투주맙": "sacituzumab",
    "타목시펜": "tamoxifen",
    "레트로졸": "letrozole",
    "아나스트로졸": "anastrozole",
    "카페시타빈": "capecitabine",
    "페메트렉시드": "pemetrexed",
    "보르테조밉": "bortezomib",
    "레날리도마이드": "lenalidomide",
    "이매티닙": "imatinib",
    "닐로티닙": "nilotinib",
    "다사티닙": "dasatinib",
    "소라페닙": "sorafenib",
    "수니티닙": "sunitinib",
    "게피티닙": "gefitinib",
    "엘로티닙": "erlotinib",
    "오시머티닙": "osimertinib",
    "레이저티닙": "lazertinib",
    "팔보시클립": "palbociclib",
    "리보시클립": "ribociclib",
    # 면역·biologics
    "아달리무맙": "adalimumab",
    "에타너셉트": "etanercept",
    "인플릭시맙": "infliximab",
    "우스테키누맙": "ustekinumab",
    "세쿠키누맙": "secukinumab",
    "익세키주맙": "ixekizumab",
    "토실리주맙": "tocilizumab",
    "오말리주맙": "omalizumab",
    "두필루맙": "dupilumab",
    "리산키주맙": "risankizumab",
    "데노수맙": "denosumab",
    "우파다시티닙": "upadacitinib",
    "바리시티닙": "baricitinib",
    "토파시티닙": "tofacitinib",
}

# 한국어 브랜드명 → 영어 generic
_BRAND_KO_EN: dict[str, str] = {
    # 일반·만성
    "타이레놀": "acetaminophen",
    "부루펜": "ibuprofen",
    "낙센": "naproxen",
    "쎄레브렉스": "celecoxib",
    "알레그라": "fexofenadine",
    "지르텍": "cetirizine",
    "클라리틴": "loratadine",
    "가스터": "famotidine",
    "넥시움": "esomeprazole",
    "란스톤": "lansoprazole",
    "오메드": "omeprazole",
    "판토록": "pantoprazole",
    "무코펙트": "ambroxol",
    "노바스크": "amlodipine",
    "코자": "losartan",
    "딜라트렌": "carvedilol",
    "리피토": "atorvastatin",
    "크레스토": "rosuvastatin",
    "라식스": "furosemide",
    "다이아벡스": "metformin",
    "아마릴": "glimepiride",
    "씬지로이드": "levothyroxine",
    "졸로프트": "sertraline",
    "렉사프로": "escitalopram",
    "자낙스": "alprazolam",
    "스틸녹스": "zolpidem",
    "콘서타": "methylphenidate",
    "애더럴": "amphetamine",
    "타미플루": "oseltamivir",
    "프라닥사": "dabigatran",
    "엘리퀴스": "apixaban",
    "자렐토": "rivaroxaban",
    # 신약
    "위고비": "semaglutide",
    "오젬픽": "semaglutide",
    "마운자로": "tirzepatide",
    "젭바운드": "tirzepatide",
    "삭센다": "liraglutide",
    "트루리시티": "dulaglutide",
    "자디앙": "empagliflozin",
    "포시가": "dapagliflozin",
    "자누비아": "sitagliptin",
    "트라젠타": "linagliptin",
    "엔트레스토": "sacubitril",
    "팍스로비드": "nirmatrelvir",
    "라게브리오": "molnupiravir",
    "베클루리": "remdesivir",
    "빅타비": "bictegravir",
    "레켐비": "lecanemab",
    "타그리소": "osimertinib",
    "렉라자": "lazertinib",
    "입랜스": "palbociclib",
    "키스칼리": "ribociclib",
    "트로델비": "sacituzumab",
    # 특수·항암·biologics
    "휴미라": "adalimumab",
    "엔브렐": "etanercept",
    "레미케이드": "infliximab",
    "스텔라라": "ustekinumab",
    "코센틱스": "secukinumab",
    "탈츠": "ixekizumab",
    "악템라": "tocilizumab",
    "졸레어": "omalizumab",
    "듀피젠트": "dupilumab",
    "스카이리치": "risankizumab",
    "프롤리아": "denosumab",
    "린버크": "upadacitinib",
    "올루미언트": "baricitinib",
    "젤잔즈": "tofacitinib",
    "키트루다": "pembrolizumab",
    "옵디보": "nivolumab",
    "여보이": "ipilimumab",
    "허셉틴": "trastuzumab",
    "퍼제타": "pertuzumab",
    "아바스틴": "bevacizumab",
    "리툭산": "rituximab",
    "얼비툭스": "cetuximab",
    "글리벡": "imatinib",
    "타시그나": "nilotinib",
    "스프라이셀": "dasatinib",
    "젤로다": "capecitabine",
    "페마라": "letrozole",
    "아리미덱스": "anastrozole",
    "넥사바": "sorafenib",
    "수텐트": "sunitinib",
    "이레사": "gefitinib",
    "타쎄바": "erlotinib",
    "알림타": "pemetrexed",
    "벨케이드": "bortezomib",
    "레블리미드": "lenalidomide",
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
