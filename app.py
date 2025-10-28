# -*- coding: utf-8 -*-
# ETF Backfill Portfolio Visualizer — Result-first + Extended Metrics & Cash Flows
# - Result-first UX after run
# - Presets load → sidebar immediate update
# - Added inputs: Initial Amount, Monthly Contribution, Dividend Reinvest (Adj Close)
# - Added metrics: Period, Longest Underwater, Sortino, Sharpe, CAGR/Longest UW, Start/End Balance
# - Composition pie & table

from __future__ import annotations
import os
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===== Korean font setup (matplotlib & system-agnostic) =====
from matplotlib import font_manager, rcParams
# ----- 한글 폰트 설정 -----
from matplotlib import font_manager, rcParams
import os

def set_korean_font():
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    else:
        rcParams["font.family"] = "Malgun Gothic, Apple SD Gothic Neo, NanumGothic, DejaVu Sans"
    rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

set_korean_font()

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
    from matplotlib import font_manager, rcParams
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
    "IEF":  ProxySpec("YF", "IEF", "U.S. Treasury 7-10y (ETF proxy)"),
    "VGLT": ProxySpec("YF", "VGLT", "U.S. Treasury Long (ETF proxy)"),
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
def to_monthly(s: pd.Series) -> pd.Series:
    return s.resample("M").last()

def drawdown_series(idx_series: pd.Series) -> Tuple[pd.Series, int]:
    """Return drawdown series and longest underwater duration in months."""
    if idx_series.empty:
        return pd.Series(dtype=float), 0
    peak = idx_series.cummax()
    dd = idx_series / peak - 1.0
    # longest underwater (below 0)
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
    out.update({"CAGR": cagr, "Vol": vol, "MDD": mdd, "Sharpe": sharpe, "Sortino": sortino,
                "UW_months": uw_months, "UW_years": uw_years, "CAGR_div_UW": cagr_div_uw})
    return out

def simulate_value_from_index(port_index: pd.Series, initial_amount: float, monthly_contrib: float) -> pd.Series:
    """
    Simulate portfolio value from index(=100 base) using monthly compounding.
    Contribution happens at each month-end AFTER growth for the month.
    """
    if port_index.empty:
        return pd.Series(dtype=float)
    m_idx = to_monthly(port_index)  # level series (e.g., 120, 130)
    m_idx = m_idx / m_idx.iloc[0]  # normalize to 1.0
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

def fmt_pct(x: float) -> str:
    return "—" if (x is None or not np.isfinite(x)) else f"{x*100:,.2f}%"

def render_line_chart_matplotlib(series, title="포트폴리오 지수 (=100 기준)"):
    import matplotlib.pyplot as plt
    import streamlit as st

    # 그래프 없으면 안내
    if series is None or series.empty:
        st.warning("표시할 결과가 없습니다.")
        return

    # 그래프 그리기
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("지수")
    ax.grid(True, alpha=0.3)

    # Streamlit 화면에 표시
    st.pyplot(fig)


def render_results(port_series: Optional[pd.Series], metrics: Optional[dict], comp_df: Optional[pd.DataFrame],
                   start_dt: date, end_dt: date, value_series: Optional[pd.Series]):
    # 기간
    st.markdown(f"**기간:** {start_dt.isoformat()} → {end_dt.isoformat()}  "
                f"(총 {(end_dt - start_dt).days}일)")
    st.subheader("포트폴리오 지수 (=100 기준)")
    if port_series is None or port_series.empty:
        st.warning("표시할 결과가 없습니다.")
        return
    render_line_chart_matplotlib(port_series, title="포트폴리오 지수 (=100 기준)")

    # 지표
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", fmt_pct(metrics["CAGR"]))
        col2.metric("변동성(연)", fmt_pct(metrics["Vol"]))
        col3.metric("최대낙폭", fmt_pct(metrics["MDD"]))
        col4.metric("Sharpe", "—" if not np.isfinite(metrics["Sharpe"]) else f"{metrics['Sharpe']:.2f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Sortino", "—" if not np.isfinite(metrics["Sortino"]) else f"{metrics['Sortino']:.2f}")
        col6.metric("최장 UW 기간", f"{metrics['UW_months']}개월 ({metrics['UW_years']:.2f}년)" if metrics["UW_months"] else "—")
        col7.metric("CAGR / 최장 UW", "—" if not np.isfinite(metrics["CAGR_div_UW"]) else f"{metrics['CAGR_div_UW']:.2f}")

    # Start/End Balance
    if value_series is not None and not value_series.empty:
        start_bal = float(value_series.iloc[0])
        end_bal = float(value_series.iloc[-1])
        st.markdown("---")
        b1, b2 = st.columns(2)
        b1.metric("Start Balance", f"{start_bal:,.0f}")
        b2.metric("End Balance", f"{end_bal:,.0f}")

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

# =============================
# Sidebar — Portfolio Editor & Run Options
# =============================
st.sidebar.header("1) 포트폴리오 구성")

def _empty_rows(n=4):
    return pd.DataFrame({"티커": ["" for _ in range(n)], "비율 (%)": [0.0 for _ in range(n)]})

if "portfolio_rows" not in st.session_state:
    st.session_state["portfolio_rows"] = _empty_rows()

# result-first state
st.session_state.setdefault("backtest_started", False)
st.session_state.setdefault("port_series", None)
st.session_state.setdefault("port_metrics", None)
st.session_state.setdefault("port_comp", None)
st.session_state.setdefault("port_value_series", None)

# ensure some empty rows
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

proxy_table, proxy_map = build_proxy_table_with_autofix(base_df)

def _append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    total = float(pd.to_numeric(d["비율 (%)"], errors="coerce").fillna(0).sum())
    d = pd.concat([d, pd.DataFrame({"티커": ["합계"], "비율 (%)": [total]})], ignore_index=True)
    return d

editor_df = _append_total_row(base_df)

label_map = {r.ETF: r.Label for _, r in proxy_table.iterrows()}
def _label_for(t):
    t = str(t).upper().strip()
    if t == "합계": return "—"
    return label_map.get(t, "알 수 없음")

editor_df["추종지수(자동)"] = editor_df["티커"].apply(_label_for)

edited_df_out = st.sidebar.data_editor(
    editor_df,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "티커": st.column_config.TextColumn("티커", help="예: QQQ, IEF, IAU, BCI"),
        "비율 (%)": st.column_config.NumberColumn("비율 (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f %%"),
        "추종지수(자동)": st.column_config.TextColumn("추종지수(자동)", help="자동 매핑 라벨", disabled=True),
    },
    disabled=["추종지수(자동)"],
)

st.session_state["portfolio_rows"] = edited_df_out.iloc[:-1][["티커", "비율 (%)"]]

st.sidebar.header("2) 기간 및 현금흐름 설정")
colA, colB = st.sidebar.columns(2)
with colA:
    start_date = st.date_input("시작일", value=date(1990,1,1))
with colB:
    end_date = st.date_input("종료일", value=date.today())

reinvest = st.sidebar.checkbox("배당 재투자(Adj Close 사용)", value=True, help="체크 해제 시 Close 사용 (총수익 제외)")
initial_amount = st.sidebar.number_input("초기 금액", min_value=0, value=10_000_000, step=100_000)
monthly_contrib = st.sidebar.number_input("월 납입액", min_value=0, value=0, step=100_00)

st.sidebar.header("3) 실행")
run_bt = st.sidebar.button("백테스트 실행", type="primary")
reset_bt = st.sidebar.button("초기화(처음 화면으로)", type="secondary")

if reset_bt:
    st.session_state.update({
        "backtest_started": False,
        "port_series": None,
        "port_metrics": None,
        "port_comp": None,
        "port_value_series": None,
    })

# =============================
# Backtest Execution
# =============================
main_tab1, main_tab2 = st.tabs(["📈 결과", "🧪 매핑 리포트"])

if run_bt:
    with st.spinner("데이터 로딩 및 백테스트 중..."):
        dfp = st.session_state["portfolio_rows"].copy()
        # Clean rows
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

                # Build each hybrid series
                series_map = {}
                for t, w in zip(dfp["티커"].tolist(), weights.tolist()):
                    proxy_sym = resolve_proxy_ticker(t, proxy_map)
                    hy = build_hybrid_series_from_proxy(
                        t, proxy_sym, start=start_date.isoformat(), auto_adjust=reinvest
                    )
                    series_map[t] = hy

                # Align & combine to portfolio index
                all_idx = None
                for s in series_map.values():
                    all_idx = s.index if all_idx is None else all_idx.union(s.index)
                all_idx = pd.DatetimeIndex(sorted(all_idx))
                parts = []
                for t, w in zip(dfp["티커"].tolist(), weights.tolist()):
                    s = series_map[t].reindex(all_idx).ffill()
                    s = s / s.iloc[0] * 100.0
                    parts.append(s * w)
                port = pd.concat(parts, axis=1).sum(axis=1).dropna()
                port = port.loc[(port.index >= pd.to_datetime(start_date)) & (port.index <= pd.to_datetime(end_date))]
                m = perf_metrics(port)

                # Simulate cash flows for balances (using monthly compounding)
                value_series = simulate_value_from_index(port, initial_amount, monthly_contrib)

                # Save to state
                st.session_state.update({
                    "backtest_started": True,
                    "port_series": port,
                    "port_metrics": m,
                    "port_comp": comp_df,
                    "port_value_series": value_series,
                })

    with main_tab2:
        rep, _ = audit_and_autofix_proxies(dfp["티커"].tolist(), BASE_PROXY_MAP)
        st.dataframe(rep, use_container_width=True)

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
        st.dataframe(proxy_table, use_container_width=True)

st.markdown("---")
st.caption("ⓘ 참고: ‘배당 재투자’ 옵션을 켜면 Adjusted Close(총수익 근사)를 사용합니다. 끄면 Close(가격수익) 기준입니다. ‘월 납입액’은 매월 말 리밸런싱 없이 단순 적립으로 가정합니다.")


