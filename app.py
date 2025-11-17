# -*- coding: utf-8 -*-
# ETF Backfill Portfolio Visualizer — Result-first + Extended Metrics & Cash Flows + Benchmark Compare
# - Result-first UX after run
# - Presets load → sidebar immediate update
# - Added inputs: Initial Amount, Monthly Contribution, Dividend Reinvest (Adj Close)
# - Added metrics: Period, Longest Underwater, Sortino, Sharpe, CAGR/Longest UW, Start/End Balance
# - Composition pie & table
# - Benchmark selection (S&P500 / 60-40 / All Weather / Global Market / None)
# - NEW (2025-10-30):
#     1) 벤치마크 기간을 ‘입력한 포트폴리오’의 기간과 정확히 동일하게 트리밍
#     2) ‘누적 수익률 비교’ 그래프의 y축을 지수(=100) 대신 실제 ‘금액(초기금액+월납입 반영)’으로 표시

from __future__ import annotations
import os
import math
import re
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager, rcParams
from matplotlib import gridspec

# ===== Korean font setup (matplotlib & system-agnostic) =====

def set_korean_font():
    # 1) 앱 로컬에 폰트를 넣었을 경우(권장): ./fonts/NanumGothic.ttf
    local_candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf"),
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "NanumGothic.ttf"),
    ]
    # 2) OS 기본 폰트 경로들
    system_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",           # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",    # Linux (Noto)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",                # macOS
        "C:/Windows/Fonts/malgun.ttf",                               # Windows
    ]
    candidates = local_candidates + system_candidates

    chosen = None
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                chosen = font_manager.FontProperties(fname=p).get_name()
                break
            except Exception:
                continue

    # 폰트를 못 찾았어도 앱이 죽지 않도록 처리
    if chosen:
        rcParams["font.family"] = chosen
    else:
        # 최소한 한글 포함 가능성이 있는 패밀리 지정 시도
        rcParams["font.family"] = "NanumGothic, Apple SD Gothic Neo, Malgun Gothic, Noto Sans CJK KR, DejaVu Sans"

    rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

set_korean_font()

# Optional: Korean font for matplotlib (best-effort)
try:
    CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in CANDIDATES:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="ETF Backfill Portfolio Visualizer", layout="wide")

# =============================
# Data & Mapping Layer
# =============================
@dataclass
class ProxySpec:
    source: str
    series: str
    name: str
    transform: str = "identity"

BASE_PROXY_MAP: Dict[str, ProxySpec] = {
    "QQQ":  ProxySpec("YF", "^NDX", "NASDAQ-100 Index"),
    "SPY":  ProxySpec("YF", "^GSPC", "S&P 500 Index"),
    "VTI":  ProxySpec("YF", "VTI", "Total U.S. Stock (ETF proxy)"),
    "VEA":  ProxySpec("YF", "VEA", "Dev ex-US (ETF proxy)"),
    "VWO":  ProxySpec("YF", "VWO", "EM Equities (ETF proxy)"),
    "VNQ":  ProxySpec("YF", "VNQ", "U.S. REITs (ETF proxy)"),
    "IEF":  ProxySpec("YF", "IEF", "U.S. Treasury 7-10y (ETF)"),
    "VGLT": ProxySpec("YF", "VGLT", "U.S. Treasury Long (ETF)"),
    "BND":  ProxySpec("YF", "AGG", "U.S. Aggregate (AGG as proxy)"),
    "AGG":  ProxySpec("YF", "AGG", "U.S. Aggregate Bonds"),
    "BIL":  ProxySpec("YF", "BIL", "1-3M T-Bill"),
    # Commodities & Gold
    "IAU":  ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "GLD":  ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "IAU.M":ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "IAUUSD":ProxySpec("YF", "XAUUSD=X","Spot Gold USD"),
    "DBC":  ProxySpec("YF", "DBC",     "Broad Commodities (ETF)"),
    "BCI":  ProxySpec("YF", "^SPGSCI", "S&P GSCI Index (broad commodity)"),
    "VT":   ProxySpec("YF", "VT",      "Global Market (ETF)"),
}

@st.cache_data(show_spinner=False)
def yf_download(symbol: str, start: str = "1970-01-01", end: Optional[str] = None, auto_adjust: bool = True) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    col = "Adj Close" if auto_adjust and "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[-1])
    s = df[col].dropna()
    s.name = symbol
    return s

@st.cache_data(show_spinner=False)
def yf_info(ticker: str) -> dict:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "longName": info.get("longName"),
            "shortName": info.get("shortName"),
            "underlyingIndex": info.get("underlyingIndex"),
            "category": info.get("category"),
            "fundFamily": info.get("fundFamily"),
        }
    except Exception:
        return {}

_RULES: List[Tuple[re.Pattern, ProxySpec]] = [
    (re.compile(r"gold|금", re.I), ProxySpec("YF", "GLD", "Gold proxy via GLD")),
    (re.compile(r"commodity|원자재|gsci|bcom", re.I), ProxySpec("YF", "^SPGSCI", "S&P GSCI Index")),
    (re.compile(r"nasdaq.*100", re.I), ProxySpec("YF", "^NDX", "NASDAQ-100")),
    (re.compile(r"s&p.*500|sp\s*500", re.I), ProxySpec("YF", "^GSPC", "S&P 500")),
    (re.compile(r"7\-10|intermediate.*treasury|중기국채", re.I), ProxySpec("YF", "IEF", "U.S. Treasury 7-10y")),
    (re.compile(r"tips|물가연동", re.I), ProxySpec("YF", "TIP", "U.S. TIPS (ETF)")),
    (re.compile(r"reit|리츠", re.I), ProxySpec("YF", "VNQ", "U.S. REITs (ETF)")),
]

@st.cache_data(show_spinner=False)
def audit_and_autofix_proxies(etfs: List[str], base_map: Dict[str, ProxySpec]) -> Tuple[pd.DataFrame, Dict[str, ProxySpec]]:
    pmap = dict(base_map)
    rows = []
    for t in etfs:
        key = t.upper().strip()
        if key in pmap:
            rows.append({"티커": key, "상태": "OK", "제안": f"{pmap[key].source}:{pmap[key].series}"})
            continue
        info = yf_info(key)
        text = " ".join([str(v) for v in info.values() if v]).lower()
        suggestion = None
        for pat, spec in _RULES:
            if pat.search(text) or pat.search(key):
                suggestion = spec
                break
        if suggestion is not None:
            pmap[key] = suggestion
            rows.append({"티커": key, "상태": "AUTO_MAPPED", "제안": f"{suggestion.source}:{suggestion.series}"})
        else:
            rows.append({"티커": key, "상태": "NEEDS_MANUAL", "제안": "(없음)"})
    return pd.DataFrame(rows), pmap

def resolve_proxy_ticker(ticker: str, proxy_map: Dict[str, ProxySpec]) -> str:
    t = ticker.upper().strip()
    spec = proxy_map.get(t)
    if spec:
        return spec.series
    if t in {"IAU", "GLD"}: return "GLD"
    if t in {"BCI", "DBC"}: return "^SPGSCI"
    return ""

def build_hybrid_series_from_proxy(etf_ticker: str, proxy_ticker: str, start: str = "1970-01-01", auto_adjust: bool = True) -> pd.Series:
    s_etf = yf_download(etf_ticker, start=start, auto_adjust=auto_adjust)
    if not proxy_ticker:
        s_etf.name = f"HYBRID_{etf_ticker}"
        return s_etf
    s_proxy = yf_download(proxy_ticker, start=start, auto_adjust=auto_adjust)
    if s_etf.empty and s_proxy.empty:
        return pd.Series(dtype=float, name=f"HYBRID_{etf_ticker}")
    idx = pd.date_range(
        start=min([x.index.min() for x in [s_etf, s_proxy] if not x.empty]),
        end=max([x.index.max() for x in [s_etf, s_proxy] if not x.empty]),
        freq="B",
    )
    e = s_etf.reindex(idx).ffill()
    p = s_proxy.reindex(idx).ffill()
    overlap = pd.concat([e, p], axis=1).dropna()
    if overlap.empty:
        scaled_p = p
    else:
        x = overlap.iloc[:, 1].values
        y = overlap.iloc[:, 0].values
        denom = float(x @ x) if np.isfinite(x @ x) and (x @ x) != 0 else 1.0
        a = float((x @ y) / denom)
        scaled_p = p * a
    cutoff = e.first_valid_index()
    hybrid = scaled_p.copy()
    if cutoff is not None:
        hybrid.loc[cutoff:] = e.loc[cutoff:]
    hybrid.name = f"HYBRID_{etf_ticker}"
    return hybrid.dropna()

# =============================
# Metrics & Helpers
# =============================

def _as_series(obj, name: str) -> pd.Series:
    """Safely convert DataFrame/Series to a 1D Series and set name."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0]
        else:
            try:
                s = obj.squeeze()
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
            except Exception:
                s = obj.iloc[:, 0]
    else:
        s = obj
    try:
        s.name = name
    except Exception:
        pass
    return s

def to_monthly(s: pd.Series) -> pd.Series:
    return s.resample("M").last()

def drawdown_series(idx_series: pd.Series) -> Tuple[pd.Series, int]:
    """Return drawdown series and longest underwater duration in months."""
    if idx_series.empty:
        return pd.Series(dtype=float), 0
    peak = idx_series.cummax()
    dd = idx_series / peak - 1.0
    underwater = dd < 0
    longest = curr = 0
    for flag in underwater.astype(int):
        if flag == 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    return dd, int(longest)

def perf_metrics(series: pd.Series) -> dict:
    """Compute performance metrics using monthly data."""
    out = {"CAGR": np.nan, "Vol": np.nan, "MDD": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
           "UW_months": 0, "UW_years": np.nan, "CAGR_div_UW": np.nan}
    if series.empty:
        return out
    idx = series / series.iloc[0] * 100.0
    m_idx = to_monthly(idx)
    rets = m_idx.pct_change().dropna()
    if rets.empty:
        return out

    years = (m_idx.index[-1] - m_idx.index[0]).days / 365.25
    cagr = (m_idx.iloc[-1] / m_idx.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
    vol = rets.std() * math.sqrt(12)
    dd, uw_months = drawdown_series(m_idx)
    mdd = dd.min() if not dd.empty else np.nan

    # Sharpe (rf=0)
    mean_ann = rets.mean() * 12
    sharpe = (mean_ann / vol) if vol and np.isfinite(vol) and vol != 0 else np.nan

    # Sortino (rf=0): downside stdev
    downside = rets[rets < 0]
    ddv = downside.std() * math.sqrt(12) if not downside.empty else np.nan
    sortino = (mean_ann / ddv) if ddv and np.isfinite(ddv) and ddv != 0 else np.nan

    uw_years = uw_months / 12.0
    cagr_div_uw = (cagr / uw_years) if uw_years and uw_years > 0 else np.nan

    out.update({
        "CAGR": cagr,
        "Vol": vol,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "UW_months": uw_months,
        "UW_years": uw_years,
        "CAGR_div_UW": cagr_div_uw,   # ← 여기 변수명 통일!
    })
    return out


def simulate_value_from_index(port_index: pd.Series, initial_amount: float, monthly_contrib: float) -> pd.Series:
    """
    Simulate portfolio value from index(=100 base) using monthly compounding.
    Contribution happens at each month-end AFTER growth for the month.
    """
    if port_index.empty:
        return pd.Series(dtype=float)
    m_idx = to_monthly(port_index)
    m_idx = m_idx / m_idx.iloc[0]
    vals = []
    balance = float(initial_amount)
    prev = m_idx.iloc[0]
    vals.append(balance)
    for t, level in zip(m_idx.index[1:], m_idx.iloc[1:]):
        growth = (level / prev)
        balance = balance * float(growth) + float(monthly_contrib)
        vals.append(balance)
        prev = level
    value_series = pd.Series(vals, index=m_idx.index, name="PortfolioValue")
    return value_series

def fmt_pct(x: float) -> str:
    return "—" if (x is None or not np.isfinite(x)) else f"{x*100:,.2f}%"

# ===== NEW: generic builder (reused for Portfolio & Benchmark) =====

def build_index_from_assets(
    tickers: List[str],
    weights: List[float],
    proxy_map: Dict[str, ProxySpec],
    start_date: date,
    end_date: date,
    reinvest: bool
) -> pd.Series:
    if not tickers or not weights or sum(weights) == 0:
        return pd.Series(dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    series_map = {}
    for t, w in zip(tickers, weights.tolist()):
        proxy_sym = resolve_proxy_ticker(t, proxy_map)
        hy = build_hybrid_series_from_proxy(
            t, proxy_sym, start=start_date.isoformat(), auto_adjust=reinvest
        )
        series_map[t] = hy

    all_idx = None
    for s in series_map.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    if all_idx is None:
        return pd.Series(dtype=float)
    all_idx = pd.DatetimeIndex(sorted(all_idx))
    parts = []
    for t, w in zip(tickers, weights.tolist()):
        s = series_map[t].reindex(all_idx).ffill()
        s = s / s.iloc[0] * 100.0
        parts.append(s * w)
    out = pd.concat(parts, axis=1).sum(axis=1).dropna()
    out = out.loc[(out.index >= pd.to_datetime(start_date)) & (out.index <= pd.to_datetime(end_date))]
    return out

def build_index_from_assets_with_rebal(
    tickers: List[str],
    weights: List[float],
    proxy_map: Dict[str, ProxySpec],
    start_date: date,
    end_date: date,
    reinvest: bool,
    rebalance: str = "NONE",  # NONE, MONTHLY, QUARTERLY, SEMI, ANNUAL
) -> pd.Series:
    """Build a monthly index (=100 base) with periodic rebalancing of *target* weights."""
    if not tickers or not weights or sum(weights) == 0:
        return pd.Series(dtype=float)
    w_target = np.array(weights, dtype=float)
    w_target = w_target / w_target.sum()

    price_map = {}
    for t in tickers:
        proxy_sym = resolve_proxy_ticker(t, proxy_map)
        hy = build_hybrid_series_from_proxy(t, proxy_sym, start=start_date.isoformat(), auto_adjust=reinvest)
        price_map[t] = hy

    mpx = []
    for t in tickers:
        s = _as_series(to_monthly(price_map[t]), t)
        mpx.append(s)
    if not mpx:
        return pd.Series(dtype=float)
    mpx = pd.concat(mpx, axis=1).dropna(how="any")
    mpx = mpx.loc[(mpx.index >= pd.to_datetime(start_date)) & (mpx.index <= pd.to_datetime(end_date))]
    if mpx.shape[0] < 2:
        return (mpx.iloc[:,0] / mpx.iloc[0,0] * 100.0)

    rets = mpx.pct_change().dropna()

    def is_rebalance_month(ts: pd.Timestamp) -> bool:
        if rebalance == "NONE":
            return False
        m = ts.month
        if rebalance == "MONTHLY":
            return True
        if rebalance == "QUARTERLY":
            return m in {3,6,9,12}
        if rebalance == "SEMI":
            return m in {6,12}
        if rebalance == "ANNUAL":
            return m == 12
        return False

    w = w_target.copy()
    idx_level = [100.0]
    for i, (ts, row) in enumerate(rets.iterrows(), start=1):
        port_ret = float(np.nansum(w * row.values))
        idx_level.append(idx_level[-1] * (1.0 + port_ret))
        w = w * (1.0 + row.values)
        tw = np.nansum(w)
        w = w / tw if tw and np.isfinite(tw) else w
        if is_rebalance_month(ts):
            w = w_target.copy()

    out = pd.Series(idx_level, index=mpx.index, name="PortfolioIndex")
    return out

# =============================
# UI helpers (Intro/Presets; Result)
# =============================

def render_intro():
    st.title("ETF 백필 포트폴리오 비주얼라이저")
    st.caption("ETF 상장 전 기간까지 추종지수로 백테스트하는 웹앱입니다. (기간: 자동 최대)")
    st.markdown("---")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("🧭 처음 오셨나요?")
        st.write(
            """
            이 웹앱은 **ETF 상장 이전 구간까지** 지수/프록시를 활용해 **하이브리드 시리즈**를 만들고,
            포트폴리오 성과를 쉽게 비교할 수 있도록 돕습니다.

            - **분산투자**: 서로 다른 자산을 섞어 위험을 낮추고 안정적 성과를 추구
            - **하이브리드 백필**: 상장 이전은 프록시 지수, 상장 이후는 실제 ETF로 이어붙이기
            - **리밸런싱**: 정기적으로 비중 복원(선택 사항)
            """
        )
    with right:
        st.info("Tip: 좌측 사이드바에서 티커와 비중을 입력하고 ‘백테스트 실행’을 눌러보세요.")
    st.markdown("---")

def render_featured_portfolios():
    PRESETS = {
        "60:40 포트폴리오": {
            "desc": "성장(주식)+안정(채권)의 기본형",
            "composition": [
                {"티커": "SPY", "자산": "미국 주식", "비중(%)": 60},
                {"티커": "BND", "자산": "미국 종합채권", "비중(%)": 40},
            ],
        },
        "올웨더 포트폴리오": {
            "desc": "레이 달리오식 리스크 균형",
            "composition": [
                {"티커": "VTI",  "자산": "미국 주식",       "비중(%)": 30},
                {"티커": "VGLT", "자산": "미국 장기국채",   "비중(%)": 40},
                {"티커": "IEF",  "자산": "미국 중기국채",   "비중(%)": 15},
                {"티커": "IAU",  "자산": "금",           "비중(%)": 7.5},
                {"티커": "DBC",  "자산": "원자재",       "비중(%)": 7.5},
            ],
        },
        "GAA 포트폴리오": {
            "desc": "글로벌 광범위 분산",
            "composition": [
                {"티커": "VTI", "자산": "미국 주식",          "비중(%)": 10},
                {"티커": "VEA", "자산": "선진국(미국 제외) 주식", "비중(%)": 10},
                {"티커": "VWO", "자산": "신흥국 주식",        "비중(%)": 10},
                {"티커": "VNQ", "자산": "REITs",            "비중(%)": 10},
                {"티커": "BND", "자산": "미국 종합채권",      "비중(%)": 20},
                {"티커": "IEF", "자산": "미국 중기국채",      "비중(%)": 10},
                {"티커": "IAU", "자산": "금",               "비중(%)": 10},
                {"티커": "DBC", "자산": "원자재",           "비중(%)": 10},
                {"티커": "BIL", "자산": "현금/단기국채",     "비중(%)": 10},
            ],
        },
    }

    st.subheader("🚀 대표 포트폴리오 비교 & 빠른 불러오기")
    for i, (name, spec) in enumerate(PRESETS.items()):
        st.markdown(f"#### 📊 {name}")
        st.caption(spec.get("desc", ""))
        dfc = pd.DataFrame(spec["composition"])
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.dataframe(dfc, hide_index=True, use_container_width=True)
            if st.button(f"이 구성 불러오기", key=f"load_{i}"):
                new_df = pd.DataFrame({
                    "티커": dfc["티커"].astype(str).str.upper().str.strip().tolist(),
                    "비율 (%)": [float(x) for x in dfc["비중(%)"].tolist()],
                })
                st.session_state["portfolio_rows"] = new_df
                st.session_state["preset_portfolio"] = {
                    "assets": new_df["티커"].tolist(),
                    "labels": dfc["자산"].tolist(),
                    "weights": new_df["비율 (%)"].tolist(),
                }
                st.success(f"‘{name}’ 구성을 사이드바에 반영했습니다.")
                st.rerun()
        with c2:
            sizes = dfc["비중(%)"].astype(float).tolist()
            labels = (dfc["자산"] + " (" + dfc["비중(%)"].astype(str) + "%)").tolist()
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        st.markdown("---")

def render_comp_pie(comp_df: pd.DataFrame):
    if comp_df is None or comp_df.empty:
        st.warning("포트폴리오 구성이 비어 있습니다.")
        return
    sizes = comp_df["비중(%)"].astype(float).tolist()
    labels = (comp_df["티커"].astype(str) + " (" + comp_df["비중(%)"].round(1).astype(str) + "%)").tolist()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

def render_line_chart_matplotlib(series, title="포트폴리오 지수 (=100 기준)"):
    if series is None or series.empty:
        st.warning("표시할 결과가 없습니다.")
        return
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("지수")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ===== NEW: SNS용 파스텔 카드 이미지 생성 함수 =====

def create_sns_image(
    value_series: Optional[pd.Series],
    bench_value_series: Optional[pd.Series],
    metrics: Optional[dict],
    comp_df: Optional[pd.DataFrame],
    start_dt: date,
    end_dt: date,
    bench_label: Optional[str] = None,
    title: str = "나의 ETF 포트폴리오",
) -> Optional[io.BytesIO]:
    """
    백테스트 결과를 파스텔 감성 SNS 카드(PNG)로 생성.
    - 누적 금액 라인 차트
    - 핵심 KPI 4개
    - 포트폴리오 구성 파이차트 + 구성 리스트
    (슬로건/카피 없음)
    """
    if value_series is None or value_series.empty:
        return None

    fig = plt.figure(figsize=(6, 9), dpi=200)
    bg_color = "#FFEFE8"
    fig.patch.set_facecolor(bg_color)

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.0, 1.2, 1.3])

    # ---------------- 1) 누적 금액 라인 차트 ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")

    ax1.plot(
        value_series.index,
        value_series.values,
        label="포트폴리오",
        linewidth=2.2,
        color="#7C8CF8",
    )
    if bench_value_series is not None and not bench_value_series.empty:
        ax1.plot(
            bench_value_series.index,
            bench_value_series.values,
            label=bench_label or "벤치마크",
            linewidth=1.8,
            linestyle="--",
            color="#B0B8C8",
            alpha=0.9,
        )

    ax1.set_title("누적 금액 추이", fontsize=12, pad=6)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    # 전체 타이틀 / 기간 / 부제
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(
        0.5, 0.952,
        f"{start_dt.isoformat()} ~ {end_dt.isoformat()}",
        ha="center", va="top", fontsize=9, color="#666666",
    )
    fig.text(
        0.5, 0.938,
        "ETF 백테스트 & 현금흐름 시뮬레이션",
        ha="center", va="top", fontsize=9,
    )

    # ---------------- 2) KPI 카드 영역 ----------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")

    if metrics:
        kpi_items = [
            ("CAGR",        metrics.get("CAGR")),
            ("변동성(연)",  metrics.get("Vol")),
            ("최대낙폭",    metrics.get("MDD")),
            ("샤프",        metrics.get("Sharpe")),
        ]
    else:
        kpi_items = [("CAGR", None), ("변동성(연)", None), ("최대낙폭", None), ("샤프", None)]

    def _fmt_kpi(name, v):
        if v is None or not np.isfinite(v):
            return "—"
        if name in {"CAGR", "변동성(연)", "최대낙폭"}:
            return f"{v*100:,.2f}%"
        if name == "샤프":
            return f"{v:.2f}"
        return str(v)

    box_colors = ["#F5D9FF", "#DFF6EA", "#FFE0D4", "#E3ECFF"]
    for i, (label, val) in enumerate(kpi_items):
        row = i // 2
        col = i % 2
        x0 = 0.05 + col * 0.48
        y0 = 0.55 - row * 0.55
        w = 0.43
        h = 0.45

        ax2.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                transform=ax2.transAxes,
                facecolor=box_colors[i],
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax2.text(
            x0 + w / 2,
            y0 + h * 0.65,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )
        ax2.text(
            x0 + w / 2,
            y0 + h * 0.32,
            _fmt_kpi(label, val),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    # ---------------- 3) 포트폴리오 구성 ----------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("white")
    ax3.axis("off")

    if comp_df is not None and not comp_df.empty:
        # 왼쪽에 파이차트
        sizes = comp_df["비중(%)"].astype(float).tolist()
        labels = comp_df["티커"].astype(str).tolist()

        ax3_pie = ax3.inset_axes([0.02, 0.05, 0.55, 0.9])
        ax3_pie.pie(
            sizes,
            labels=labels,
            startangle=90,
            autopct="%1.0f%%",
            textprops={"fontsize": 7},
        )
        ax3_pie.axis("equal")

        # 오른쪽에 구성 리스트 텍스트
        ax3.text(
            0.68, 0.78,
            "포트폴리오 구성",
            transform=ax3.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

        for i, (tkr, w) in enumerate(zip(labels, comp_df["비중(%)"].round(1))):
            y = 0.62 - i * 0.12
            ax3.text(
                0.68, y,
                f"{tkr}  {w:.1f}%",
                transform=ax3.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                color="#555555",
            )
    else:
        ax3.text(
            0.5, 0.5,
            "포트폴리오 구성 데이터 없음",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )

    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.93])

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    plt.close(fig)
    return buf

# ===== NEW: 결과 렌더러 (벤치마크 비교 포함, 금액 축) =====

def render_results(
    port_series: Optional[pd.Series],
    metrics: Optional[dict],
    comp_df: Optional[pd.DataFrame],
    start_dt: date,
    end_dt: date,
    value_series: Optional[pd.Series],
    bench_series: Optional[pd.Series] = None,
    bench_label: Optional[str] = None,
    bench_metrics: Optional[dict] = None,
    bench_value_series: Optional[pd.Series] = None,
):
    st.markdown(f"**기간:** {start_dt.isoformat()} → {end_dt.isoformat()}  "
                f"(총 {(end_dt - start_dt).days}일)")
    st.subheader("누적 금액 비교 (초기금액/월납입 반영)")

    if value_series is None or value_series.empty:
        st.warning("표시할 결과가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(value_series.index, value_series.values, label="포트폴리오", linewidth=2)
    if bench_value_series is not None and not bench_value_series.empty:
        ax.plot(
            bench_value_series.index,
            bench_value_series.values,
            label=(bench_label or "벤치마크"),
            linestyle="--",
            alpha=0.9,
        )
    ax.set_title("누적 금액 (초기금액·월납입 반영)")
    ax.set_ylabel("잔고 (원)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("성과 지표 비교표")
    if metrics is not None:
        start_bal = end_bal = np.nan
        if value_series is not None and not value_series.empty:
            start_bal = float(value_series.iloc[0])
            end_bal = float(value_series.iloc[-1])
        bench_start_bal = bench_end_bal = np.nan
        if bench_value_series is not None and not bench_value_series.empty:
            bench_start_bal = float(bench_value_series.iloc[0])
            bench_end_bal = float(bench_value_series.iloc[-1])

        rows = [
            ["CAGR",            metrics["CAGR"],            bench_metrics["CAGR"]            if bench_metrics else np.nan],
            ["Volatility",      metrics["Vol"],             bench_metrics["Vol"]             if bench_metrics else np.nan],
            ["Max Drawdown",    metrics["MDD"],             bench_metrics["MDD"]             if bench_metrics else np.nan],
            ["Sharpe",          metrics["Sharpe"],          bench_metrics["Sharpe"]          if bench_metrics else np.nan],
            ["Sortino",         metrics["Sortino"],         bench_metrics["Sortino"]         if bench_metrics else np.nan],
            ["UW (months)",     float(metrics["UW_months"]),float(bench_metrics["UW_months"]) if bench_metrics else np.nan],
            ["CAGR / UW(years)",metrics["CAGR_div_UW"],     bench_metrics["CAGR_div_UW"]     if bench_metrics else np.nan],
            ["Start Balance",   start_bal,                  bench_start_bal],
            ["End Balance",     end_bal,                    bench_end_bal],
        ]
        comp_tbl = pd.DataFrame(rows, columns=["지표", "포트폴리오", bench_label or "벤치마크"])

        pct_cols = {"CAGR", "Volatility", "Max Drawdown"}
        def _fmt(z, row_name):
            if row_name in pct_cols:
                return "—" if not np.isfinite(z) else f"{z*100:,.2f}%"
            if row_name in {"Sharpe", "Sortino", "CAGR / UW(years)"}:
                return "—" if not np.isfinite(z) else f"{z:.2f}"
            if row_name == "UW (months)":
                return "—" if not np.isfinite(z) else f"{z:.0f}개월"
            if row_name in {"Start Balance", "End Balance"}:
                return "—" if not np.isfinite(z) else f"{z:,.0f}"
            return z

        comp_tbl["포맷_포트"] = [_fmt(v, r) for r, v in zip(comp_tbl["지표"], comp_tbl["포트폴리오"])]
        comp_tbl["포맷_벤치"] = [_fmt(v, r) for r, v in zip(comp_tbl["지표"], comp_tbl[bench_label or "벤치마크"])]
        show_tbl = comp_tbl[["지표", "포맷_포트", "포맷_벤치"]].rename(columns={
            "포맷_포트": "포트폴리오", "포맷_벤치": bench_label or "벤치마크"
        })
        order = [
            "Start Balance", "End Balance", "CAGR", "Volatility", "Max Drawdown", "Sharpe", "Sortino", "CAGR / UW(years)"
        ]
        show_tbl = show_tbl.set_index("지표").reindex(order).reset_index()
        st.table(show_tbl)
    else:
        st.caption("지표가 없습니다.")

    st.markdown("---")
    st.subheader("구성 비율")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        render_comp_pie(comp_df if comp_df is not None else pd.DataFrame())
    with col2:
        if comp_df is not None and not comp_df.empty:
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
        else:
            st.caption("구성 데이터가 없습니다.")

    # === SNS용 이미지 섹션 ===
    st.markdown("---")
    st.subheader("SNS 공유용 카드 (파스텔)")
    sns_buf = create_sns_image(
        value_series=value_series,
        bench_value_series=bench_value_series,
        metrics=metrics,
        comp_df=comp_df,
        start_dt=start_dt,
        end_dt=end_dt,
        bench_label=bench_label,
        title="나의 ETF 포트폴리오",
    )
    if sns_buf is not None:
        st.image(sns_buf, caption="SNS 공유용 미리보기")
        st.download_button(
            "SNS 이미지 다운로드 (PNG)",
            data=sns_buf.getvalue(),
            file_name=f"portfolio_sns_{end_dt.isoformat()}.png",
            mime="image/png",
        )
    else:
        st.caption("SNS 이미지를 생성할 수 있는 데이터가 부족합니다.")

# =============================
# Sidebar — Portfolio Editor & Run Options
# =============================

st.sidebar.header("1) 포트폴리오 구성")

def _empty_rows(n=4):
    return pd.DataFrame({"티커": ["" for _ in range(n)], "비율 (%)": [0.0 for _ in range(n)]})

if "portfolio_rows" not in st.session_state:
    st.session_state["portfolio_rows"] = _empty_rows()

st.session_state.setdefault("backtest_started", False)
st.session_state.setdefault("port_series", None)
st.session_state.setdefault("port_metrics", None)
st.session_state.setdefault("port_comp", None)
st.session_state.setdefault("port_value_series", None)
st.session_state.setdefault("bench_series", None)
st.session_state.setdefault("bench_metrics", None)
st.session_state.setdefault("bench_label", None)
st.session_state.setdefault("bench_value_series", None)

base_df = st.session_state["portfolio_rows"]
if len(base_df) < 6:
    base_df = pd.concat([base_df, _empty_rows(6 - len(base_df))], ignore_index=True)

def build_proxy_table_with_autofix(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ProxySpec]]:
    tickers = [t for t in df_in["티커"].astype(str).str.upper().str.strip().tolist() if t]
    report, pmap = audit_and_autofix_proxies(tickers, BASE_PROXY_MAP)
    rows = []
    for t in tickers:
        spec = pmap.get(t)
        label = "알 수 없음"; proxy = ""
        if spec:
            proxy = spec.series
            label = f"{spec.name} / proxy: {proxy}"
        elif t in {"IAU", "GLD"}:
            proxy = "GLD"; label = f"Gold proxy via GLD / proxy: {proxy}"
        elif t in {"BCI", "DBC"}:
            proxy = "^SPGSCI"; label = f"S&P GSCI Index / proxy: {proxy}"
        rows.append({"ETF": t, "Label": label, "Proxy": proxy})
    return pd.DataFrame(rows), pmap

proxy_map = BASE_PROXY_MAP

def _append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    total = float(pd.to_numeric(d["비율 (%)"], errors="coerce").fillna(0).sum())
    d = pd.concat([d, pd.DataFrame({"티커": ["합계"], "비율 (%)": [total]})], ignore_index=True)
    return d

editor_df = _append_total_row(base_df)
editor_df = editor_df[["티커", "비율 (%)"]]

edited_df_out = st.sidebar.data_editor(
    editor_df,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "티커": st.column_config.TextColumn("티커", help="예: QQQ, IEF, IAU, BCI"),
        "비율 (%)": st.column_config.NumberColumn(
            "비율 (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f %%"
        ),
    }
)
st.session_state["portfolio_rows"] = edited_df_out.iloc[:-1][["티커", "비율 (%)"]]

st.sidebar.header("2) 벤치마크 선택")
BENCHMARK_PRESETS: Dict[str, Optional[Tuple[List[str], List[float]]]] = {
    "없음 (No Benchmark)": None,
    "S&P 500 (SPY)": (["SPY"], [1.0]),
    "60/40 (SPY/IEF)": (["SPY", "IEF"], [0.60, 0.40]),
    "All Weather (VTI/VGLT/IEF/IAU/DBC)": (["VTI", "VGLT", "IEF", "IAU", "DBC"], [0.30, 0.40, 0.15, 0.075, 0.075]),
    "Global Market (VT)": (["VT"], [1.0]),
}
bench_choice = st.sidebar.selectbox("📊 벤치마크", list(BENCHMARK_PRESETS.keys()), index=1)

st.sidebar.header("3) 기간 및 현금흐름 설정")
colA, colB = st.sidebar.columns(2)
with colA:
    start_date = st.date_input("시작일", value=date(1990,1,1))
with colB:
    end_date = st.date_input("종료일", value=date.today())

reinvest = st.sidebar.checkbox("배당 재투자(Adj Close 사용)", value=True, help="체크 해제 시 Close 사용 (총수익 제외)")

rebalance_choice = st.sidebar.selectbox(
    "리밸런싱 주기",
    ["없음(바이앤홀드)", "월간", "분기", "반기", "연간"],
    index=0,
)
REBAL_MAP = {
    "없음(바이앤홀드)": "NONE",
    "월간": "MONTHLY",
    "분기": "QUARTERLY",
    "반기": "SEMI",
    "연간": "ANNUAL",
}
rebalance_rule = REBAL_MAP[rebalance_choice]

initial_amount = st.sidebar.number_input("초기 금액", min_value=0, value=10_000_000, step=100_000)
monthly_contrib = st.sidebar.number_input("월 납입액", min_value=0, value=0, step=100_00)

st.sidebar.header("4) 실행")
run_bt = st.sidebar.button("백테스트 실행", type="primary")
reset_bt = st.sidebar.button("초기화(처음 화면으로)", type="secondary")

if reset_bt:
    st.session_state.update({
        "backtest_started": False,
        "port_series": None,
        "port_metrics": None,
        "port_comp": None,
        "port_value_series": None,
        "bench_series": None,
        "bench_metrics": None,
        "bench_label": None,
        "bench_value_series": None,
    })

# =============================
# Backtest Execution
# =============================

main_tab1, main_tab2 = st.tabs(["📈 결과", "🧪 매핑 리포트"])

if run_bt:
    with st.spinner("데이터 로딩 및 백테스트 중..."):
        dfp = st.session_state["portfolio_rows"].copy()
        dfp["티커"] = dfp["티커"].astype(str).str.upper().str.strip()
        dfp["비율 (%)"] = pd.to_numeric(dfp["비율 (%)"], errors="coerce").fillna(0.0)
        dfp = dfp[dfp["티커"] != ""]
        dfp = dfp[dfp["비율 (%)"] > 0]
        if dfp.empty:
            st.error("포트폴리오가 비어있습니다. 티커와 비중을 입력해주세요.")
        else:
            weights = dfp["비율 (%)"].values
            if weights.sum() == 0:
                st.error("비중 합계가 0입니다. 비중을 입력해주세요.")
            else:
                weights = weights / weights.sum()

                comp_df = pd.DataFrame({
                    "티커": dfp["티커"].tolist(),
                    "비중(%)": (weights * 100).round(2).tolist(),
                })

                rep_map, proxy_map_rt = audit_and_autofix_proxies(dfp["티커"].tolist(), BASE_PROXY_MAP)
                port = build_index_from_assets_with_rebal(
                    tickers=dfp["티커"].tolist(),
                    weights=weights.tolist(),
                    proxy_map=proxy_map_rt,
                    start_date=start_date,
                    end_date=end_date,
                    reinvest=reinvest,
                    rebalance=rebalance_rule,
                )
                m = perf_metrics(port)
                value_series = simulate_value_from_index(port, initial_amount, monthly_contrib)

                bench_label = bench_choice
                bench_series = None
                bench_metrics = None
                bench_value_series = None
                bench_spec = BENCHMARK_PRESETS.get(bench_choice)
                if bench_spec is not None:
                    b_assets, b_weights = bench_spec
                    raw_bench = build_index_from_assets_with_rebal(
                        tickers=b_assets,
                        weights=b_weights,
                        proxy_map=proxy_map_rt,
                        start_date=start_date,
                        end_date=end_date,
                        reinvest=reinvest,
                        rebalance=rebalance_rule,
                    )
                    if raw_bench is not None and not raw_bench.empty and not port.empty:
                        bench_series = raw_bench.loc[port.index.min(): port.index.max()]
                        if bench_series is not None and not bench_series.empty:
                            bench_metrics = perf_metrics(bench_series)
                            bench_value_series = simulate_value_from_index(bench_series, initial_amount, monthly_contrib)

                st.session_state.update({
                    "backtest_started": True,
                    "port_series": port,
                    "port_metrics": m,
                    "port_comp": comp_df,
                    "port_value_series": value_series,
                    "bench_series": bench_series,
                    "bench_metrics": bench_metrics,
                    "bench_label": bench_label if bench_spec is not None else None,
                    "bench_value_series": bench_value_series,
                })

    with main_tab2:
        st.subheader("매핑 리포트 (요청 시 계산)")
        if st.button("매핑 점검 실행", key="run_mapping_report"):
            rep, _ = audit_and_autofix_proxies(st.session_state["portfolio_rows"]["티커"].tolist(), BASE_PROXY_MAP)
            st.dataframe(rep, use_container_width=True)
        else:
            st.caption("입력 속도를 위해 기본적으로 매핑을 지연시킵니다. 필요할 때 위 버튼을 눌러 점검하세요.")

# =============================
# Result-first / Intro visibility control
# =============================
if st.session_state["backtest_started"]:
    with main_tab1:
        render_results(
            st.session_state["port_series"],
            st.session_state["port_metrics"],
            st.session_state.get("port_comp"),
            start_dt=start_date,
            end_dt=end_date,
            value_series=st.session_state.get("port_value_series"),
            bench_series=st.session_state.get("bench_series"),
            bench_label=st.session_state.get("bench_label"),
            bench_metrics=st.session_state.get("bench_metrics"),
            bench_value_series=st.session_state.get("bench_value_series"),
        )
    st.toggle("대표 포트폴리오 보기", value=False, key="show_presets_after_run")
    if st.session_state.get("show_presets_after_run"):
        render_featured_portfolios()
else:
    render_intro()
    render_featured_portfolios()
    with main_tab1:
        st.info("좌측 사이드바에서 포트폴리오를 입력하고 ‘백테스트 실행’을 눌러 결과를 확인하세요.")
    with main_tab2:
        st.subheader("매핑 리포트 (요청 시 계산)")
        if st.button("매핑 점검 실행", key="run_mapping_report_intro"):
            rep, _ = audit_and_autofix_proxies(st.session_state["portfolio_rows"]["티커"].tolist(), BASE_PROXY_MAP)
            st.dataframe(rep, use_container_width=True)
        else:
            st.caption("입력 중 렉을 줄이기 위해 자동 점검을 지연합니다. 필요 시 버튼을 눌러 확인하세요.")

st.markdown("---")
st.caption(
    "ⓘ 참고: '배당 재투자' 옵션을 켜면 Adjusted Close(총수익 근사)를 사용합니다. "
    "끄면 Close(가격수익) 기준입니다. "
    "'월 납입액'은 매월 말 성과 반영 후 적립으로 가정합니다. "
    "리밸런싱 주기는 선택한 주기에 맞춰 목표 비중으로 복원됩니다."
)


