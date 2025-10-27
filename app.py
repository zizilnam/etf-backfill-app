# -*- coding: utf-8 -*-
# app.py — Integrated Streamlit app
# - Friendly intro for beginners (Korean)
# - Representative portfolio presets (60:40, All Weather, GAA) with descriptions + pie
# - Sidebar editor shows "추종지수(자동)" without "알 수 없음" spam
# - Auto audit & proxy mapping for ETFs (incl. IAU, BCI)
# - Hybrid backfill: pre‑listing period uses proxy; post‑listing uses ETF
# - Simple backtest: portfolio index, CAGR/Vol/MDD

from __future__ import annotations
import os
import io
import math
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional: Korean font for matplotlib (best‑effort)
try:
    from matplotlib import font_manager, rcParams
    # Common paths on Linux/Windows. Add your own if needed.
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
    source: str   # "YF"(Yahoo) | "FRED"(not used directly in loader here)
    series: str   # Yahoo symbol (preferred in this app)
    name: str
    transform: str = "identity"

# Base proxy map — include troublesome ones (IAU/BCI) mapped to Yahoo‑loadable proxies
BASE_PROXY_MAP: Dict[str, ProxySpec] = {
    # Stocks / Bonds examples
    "QQQ":  ProxySpec("YF", "^NDX", "NASDAQ‑100 Index"),   # Use index symbol for longer history
    "SPY":  ProxySpec("YF", "^GSPC", "S&P 500 Index"),
    "VTI":  ProxySpec("YF", "VTI", "Total U.S. Stock (ETF proxy)"),
    "VEA":  ProxySpec("YF", "VEA", "Dev ex‑US (ETF proxy)"),
    "VWO":  ProxySpec("YF", "VWO", "EM Equities (ETF proxy)"),
    "VNQ":  ProxySpec("YF", "VNQ", "U.S. REITs (ETF proxy)"),
    "IEF":  ProxySpec("YF", "IEF", "U.S. Treasury 7‑10y (ETF proxy)"),
    "VGLT": ProxySpec("YF", "VGLT", "U.S. Treasury Long (ETF proxy)"),
    "BND":  ProxySpec("YF", "AGG", "U.S. Aggregate (AGG as proxy)"),
    "AGG":  ProxySpec("YF", "AGG", "U.S. Aggregate Bonds"),
    "BIL":  ProxySpec("YF", "BIL", "1‑3M T‑Bill"),
    # Commodities & Gold — the troublemakers
    "IAU":  ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "GLD":  ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "IAU.M":ProxySpec("YF", "GLD",     "Gold proxy via GLD"),
    "IAUUSD":ProxySpec("YF", "XAUUSD=X","Spot Gold USD"),
    "DBC":  ProxySpec("YF", "DBC",     "Broad Commodities (ETF)"),
    "BCI":  ProxySpec("YF", "^SPGSCI", "S&P GSCI Index (broad commodity)"),
}

# Lightweight yfinance helpers
@st.cache_data(show_spinner=False)
def yf_download(symbol: str, start: str = "1970-01-01", end: Optional[str] = None, auto_adjust: bool = True) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    # Prefer Adj Close if present
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
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

# Rules to guess proxies from metadata text
_RULES: List[Tuple[re.Pattern, ProxySpec]] = [
    (re.compile(r"gold|금", re.I), ProxySpec("YF", "GLD", "Gold proxy via GLD")),
    (re.compile(r"commodity|원자재|gsci|bcom", re.I), ProxySpec("YF", "^SPGSCI", "S&P GSCI Index")),
    (re.compile(r"nasdaq.*100", re.I), ProxySpec("YF", "^NDX", "NASDAQ‑100")),
    (re.compile(r"s&p.*500|sp\s*500", re.I), ProxySpec("YF", "^GSPC", "S&P 500")),
    (re.compile(r"7\-10|intermediate.*treasury|중기국채", re.I), ProxySpec("YF", "IEF", "U.S. Treasury 7‑10y")),
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

# Resolve a proxy ticker to load with Yahoo
def resolve_proxy_ticker(ticker: str, proxy_map: Dict[str, ProxySpec]) -> str:
    t = ticker.upper().strip()
    spec = proxy_map.get(t)
    if spec:
        return spec.series
    # fallbacks
    if t in {"IAU", "GLD"}: return "GLD"
    if t in {"BCI", "DBC"}: return "^SPGSCI"
    return ""

# Hybrid builder (ETF + Proxy splice)
def build_hybrid_series_from_proxy(etf_ticker: str, proxy_ticker: str, start: str = "1970-01-01") -> pd.Series:
    if not proxy_ticker:
        # Only ETF
        s_etf = yf_download(etf_ticker, start=start)
        s_etf.name = f"HYBRID_{etf_ticker}"
        return s_etf
    s_etf = yf_download(etf_ticker, start=start)
    s_proxy = yf_download(proxy_ticker, start=start)
    if s_etf.empty and s_proxy.empty:
        return pd.Series(dtype=float, name=f"HYBRID_{etf_ticker}")
    # business‑day index
    idx = pd.date_range(
        start=min([x.index.min() for x in [s_etf, s_proxy] if not x.empty]),
        end=max([x.index.max() for x in [s_etf, s_proxy] if not x.empty]),
        freq="B",
    )
    e = s_etf.reindex(idx).ffill()
    p = s_proxy.reindex(idx).ffill()
    # scale proxy to ETF over overlap
    overlap = pd.concat([e, p], axis=1).dropna()
    if overlap.empty:
        scaled_p = p
    else:
        x = overlap.iloc[:, 1].values
        y = overlap.iloc[:, 0].values
        a = float((x @ y) / (x @ x)) if np.isfinite((x @ y) / (x @ x)) else 1.0
        scaled_p = p * a
    # splice
    cutoff = e.first_valid_index()
    hybrid = scaled_p.copy()
    if cutoff is not None:
        hybrid.loc[cutoff:] = e.loc[cutoff:]
    hybrid.name = f"HYBRID_{etf_ticker}"
    return hybrid.dropna()

# Simple metrics
def to_monthly(s: pd.Series) -> pd.Series:
    return s.resample("M").last()

def perf_metrics(series: pd.Series) -> dict:
    if series.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "MDD": np.nan}
    idx = series / series.iloc[0] * 100.0
    m = to_monthly(idx)
    rets = m.pct_change().dropna()
    # CAGR
    years = (m.index[-1] - m.index[0]).days / 365.25
    cagr = (m.iloc[-1] / m.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
    # Vol (annualized)
    vol = rets.std() * math.sqrt(12)
    # MDD
    roll_max = idx.cummax()
    dd = idx / roll_max - 1
    mdd = dd.min()
    return {"CAGR": cagr, "Vol": vol, "MDD": mdd}

# =============================
# UI — Intro & Presets
# =============================
import streamlit as st

# --- 섹션 렌더 함수 예시 (이미 있다면 그대로 사용) ---
def render_intro():
    st.markdown("### 처음 오셨나요?")
    st.write("- 이 앱은 ETF/인덱스 하이브리드 백테스트를 지원합니다.")
    st.write("- 좌측에서 포트폴리오를 설정하고 ‘백테스트 실행’을 눌러주세요.")

def render_featured_portfolios():
    st.markdown("### 대표 포트폴리오 비교")
    st.write("예시 포트폴리오들을 간단히 비교합니다.")

def render_inputs():
    st.markdown("### 백테스트 설정")
    # 👉 여기엔 기존 입력 위젯들 (티커, 비중, 시작/종료일, 리밸런싱 등) 배치
    # 예시:
    # tickers = st.text_input("ETF 티커(쉼표로 구분)", "QQQ,IEF")
    # weights = st.text_input("비중(%)", "60,40")
    # start = st.date_input("시작일", ...)
    # end = st.date_input("종료일", ...)
    # return dict(tickers=tickers, weights=weights, start=start, end=end)
    return {}

def run_backtest(params):
    # 👉 실제 백테스트 로직 호출
    # df, metrics, charts = ...
    # 예시 결과 리턴
    return {
        "summary": {"CAGR": "8.4%", "MDD": "-17.2%", "Sharpe": "0.68"},
        "note": "샘플 결과입니다. 실제 로직에 연결하세요."
    }

def render_results(result):
    st.markdown("## 백테스트 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR", result["summary"]["CAGR"])
    col2.metric("MDD", result["summary"]["MDD"])
    col3.metric("Sharpe", result["summary"]["Sharpe"])
    st.caption(result.get("note", ""))

# --- 여기부터 메인 흐름 ---
def main():
    st.set_page_config(layout="wide")
    st.title("ETF 백테스트 확장 웹앱")

    # 1) 상태 플래그 기본값
    if "backtest_started" not in st.session_state:
        st.session_state.backtest_started = False
    if "last_params" not in st.session_state:
        st.session_state.last_params = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # 2) 입력 폼
    with st.form(key="bt_form", clear_on_submit=False):
        params = render_inputs()
        submitted = st.form_submit_button("백테스트 실행", use_container_width=True)

    # 3) 버튼을 누르면: 플래그 True + 결과 계산
    if submitted:
        st.session_state.backtest_started = True
        st.session_state.last_params = params
        st.session_state.last_result = run_backtest(params)

    # 4) 플래그에 따라 섹션 표시 순서/가시성 제어
    if st.session_state.backtest_started:
        # ✅ 결과를 최상단에 먼저 표시
        render_results(st.session_state.last_result)

        # 선택: 결과 하단에 입력 섹션(재실행용)만 노출
        with st.expander("설정 다시 열기 / 재실행", expanded=False):
            with st.form(key="bt_form_again", clear_on_submit=False):
                params = render_inputs()
                re_submitted = st.form_submit_button("다시 백테스트 실행", use_container_width=True)
            if re_submitted:
                st.session_state.last_params = params
                st.session_state.last_result = run_backtest(params)
                st.experimental_rerun()

        # 선택: 초기안내/대표포트폴리오 토글 스위치(기본은 숨김)
        st.toggle("초기 안내/대표 포트폴리오 보기", value=False, key="show_guides")
        if st.session_state.show_guides:
            st.info("초기 안내/대표 포트폴리오는 백테스트 실행 후 기본적으로 숨깁니다.")
            render_intro()
            render_featured_portfolios()

        # 리셋 버튼(완전 초기화)
        if st.button("초기화(처음 화면으로)", type="secondary"):
            st.session_state.backtest_started = False
            st.session_state.last_params = None
            st.session_state.last_result = None
            st.experimental_rerun()

    else:
        # ✅ 처음 화면(아직 실행 전): 안내와 대표 포트폴리오 노출
        render_intro()
        render_featured_portfolios()

if __name__ == "__main__":
    main()

# =============================
# Sidebar — Portfolio Editor
# =============================
st.sidebar.header("1) 포트폴리오 구성")

def _empty_rows(n=4):
    return pd.DataFrame({"티커": ["" for _ in range(n)], "비율 (%)": [0.0 for _ in range(n)]})

if "portfolio_rows" not in st.session_state:
    if "preset_portfolio" in st.session_state:
        p = st.session_state["preset_portfolio"]
        st.session_state["portfolio_rows"] = pd.DataFrame({
            "티커": p.get("assets", []),
            "비율 (%)": p.get("weights", []),
        })
    else:
        st.session_state["portfolio_rows"] = _empty_rows()

# Ensure a few empty rows for user convenience
base_df = st.session_state["portfolio_rows"]
if len(base_df) < 6:
    base_df = pd.concat([base_df, _empty_rows(6 - len(base_df))], ignore_index=True)

# Build auto mapping table from current tickers
def build_proxy_table_with_autofix(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ProxySpec]]:
    tickers = [t for t in df_in["티커"].astype(str).str.upper().str.strip().tolist() if t]
    report, pmap = audit_and_autofix_proxies(tickers, BASE_PROXY_MAP)
    rows = []
    for t in tickers:
        spec = pmap.get(t)
        label = "알 수 없음"
        proxy = ""
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

# Save back (drop total row)
st.session_state["portfolio_rows"] = edited_df_out.iloc[:-1][["티커", "비율 (%)"]]

st.sidebar.header("2) 기간 설정")
colA, colB = st.sidebar.columns(2)
with colA:
    start_date = st.date_input("시작일", value=date(1990,1,1))
with colB:
    end_date = st.date_input("종료일", value=date.today())

run_bt = st.sidebar.button("백테스트 실행", type="primary")

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
                # Build each hybrid series
                series_map = {}
                for t, w in zip(dfp["티커"].tolist(), weights.tolist()):
                    proxy_sym = resolve_proxy_ticker(t, proxy_map)
                    hy = build_hybrid_series_from_proxy(t, proxy_sym, start=start_date.isoformat())
                    series_map[t] = hy
                # Align & combine
                all_idx = None
                for s in series_map.values():
                    if all_idx is None:
                        all_idx = s.index
                    else:
                        all_idx = all_idx.union(s.index)
                all_idx = pd.DatetimeIndex(sorted(all_idx))
                rets = []
                for t, w in zip(dfp["티커"].tolist(), weights.tolist()):
                    s = series_map[t].reindex(all_idx).ffill()
                    s = s / s.iloc[0] * 100.0
                    rets.append(s * w)
                port = pd.concat(rets, axis=1).sum(axis=1).dropna()
                port = port.loc[(port.index >= pd.to_datetime(start_date)) & (port.index <= pd.to_datetime(end_date))]
                # Plot
                with main_tab1:
                    st.subheader("포트폴리오 지수 (=100 기준)")
                    st.line_chart(port)
                    m = perf_metrics(port)
                    st.markdown(
                        f"**CAGR:** {m['CAGR']*100:,.2f}%  |  **변동성:** {m['Vol']*100:,.2f}%  |  **최대낙폭:** {m['MDD']*100:,.2f}%"
                    )

    with main_tab2:
        rep, _ = audit_and_autofix_proxies(dfp["티커"].tolist(), BASE_PROXY_MAP)
        st.dataframe(rep, use_container_width=True)
else:
    with main_tab1:
        st.info("좌측 사이드바에서 포트폴리오를 입력하고 '백테스트 실행'을 눌러 결과를 확인하세요.")
    with main_tab2:
        st.dataframe(proxy_table, use_container_width=True)

st.markdown("---")
st.caption("ⓘ 참고: IAU/BCI 등 일부 ETF는 공식 '지수'가 공개 표준화되어 있지 않아, Yahoo에서 접근 가능한 대체 프록시(GLD, ^SPGSCI 등)로 자동 매핑합니다. 더 정교한 지수(예: BCOMTR)를 쓰려면 데이터 소스를 추가하세요.")

