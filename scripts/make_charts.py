"""ablation 비교 그래프 생성

- RAG 구성 종합 비교(시드620 · 80쿼리)에서 측정한 수치를 막대그래프로 렌더
- data/eval/charts/ 에 PNG 3종 저장 (모델·헤더 지표 / 언어별 R@10 / 시드·필터 개선)
- matplotlib 은 일회성으로 `uv run --with matplotlib python scripts/make_charts.py` 실행
- 수치는 data/eval/abl-*.md, results-large-*-q80*.md 측정 결과를 옮긴 것
"""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한국어 라벨용 폰트 (macOS 기본). 없으면 시스템 기본 폰트로 진행
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for _font in _FONT_CANDIDATES:
    if Path(_font).exists():
        fm.fontManager.addfont(_font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("data/eval/charts")
OUT.mkdir(parents=True, exist_ok=True)

configs = ["mpnet\n무헤더", "mpnet\n헤더", "BGE-M3\n무헤더", "BGE-M3\n헤더\n(채택)"]
R5 = [0.412, 0.438, 0.487, 0.588]
R10 = [0.500, 0.512, 0.512, 0.600]
MRR = [0.334, 0.350, 0.423, 0.577]
EN = [0.676, 0.676, 0.757, 0.784]
KO = [0.349, 0.372, 0.302, 0.442]

CHOSEN = 3  # index of 채택 구성
C1, C2, C3 = "#7eb0d5", "#5a8fbf", "#1f5e8b"
GREEN = "#6aa84f"


def label_bars(ax, bars, fmt="%.3f", size=8):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.008,
            fmt % h,
            ha="center",
            va="bottom",
            fontsize=size,
        )


# 1) 모델·헤더 지표
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(configs))
w = 0.26
b1 = ax.bar(x - w, R5, w, label="Recall@5", color=C1)
b2 = ax.bar(x, R10, w, label="Recall@10", color=C2)
b3 = ax.bar(x + w, MRR, w, label="MRR", color=C3)
ax.axvspan(CHOSEN - 0.5, CHOSEN + 0.5, color="#fff2cc", alpha=0.6, zorder=0)
for b in (b1, b2, b3):
    label_bars(ax, b)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("점수")
ax.set_ylim(0, 0.7)
ax.set_title("모델 × 헤더 구성별 검색 성능 (OFF 모드 · 시드620 · 80쿼리)")  # noqa: RUF001
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "abl_metrics.png", dpi=150)
plt.close(fig)

# 2) 언어별 R@10
fig, ax = plt.subplots(figsize=(9, 5.2))
w = 0.34
b1 = ax.bar(x - w / 2, EN, w, label="영어 (en, 37)", color="#4c72b0")
b2 = ax.bar(x + w / 2, KO, w, label="한국어 (ko, 43)", color="#dd8452")
ax.axvspan(CHOSEN - 0.5, CHOSEN + 0.5, color="#fff2cc", alpha=0.6, zorder=0)
for b in (b1, b2):
    label_bars(ax, b)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Recall@10")
ax.set_ylim(0, 0.9)
ax.set_title("언어별 Recall@10 — 조합에서 한국어가 회복 (OFF 모드)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "abl_lang.png", dpi=150)
plt.close(fig)

# 3) 시드·필터 단일 축 개선 (before/after)
fig, ax = plt.subplots(figsize=(7.5, 5.0))
groups = ["시드 확장\n(OFF R@10)", "drug_id 필터\n(ON R@10)"]
before = [0.425, 0.125]
after = [0.600, 0.625]
gx = np.arange(len(groups))
w = 0.34
b1 = ax.bar(gx - w / 2, before, w, label="개선 전", color="#bdbdbd")
b2 = ax.bar(gx + w / 2, after, w, label="개선 후", color=GREEN)
for b in (b1, b2):
    label_bars(ax, b)
ax.set_xticks(gx)
ax.set_xticklabels(groups)
ax.set_ylabel("점수")
ax.set_ylim(0, 0.75)
ax.set_title("시드 확장·필터 개선 효과 (채택 구성)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "abl_decision.png", dpi=150)
plt.close(fig)

print("saved:", *[str(p) for p in sorted(OUT.glob("*.png"))], sep="\n")
