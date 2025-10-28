# -*- coding: utf-8 -*-
"""
ETF Backfill Portfolio Visualizer — 통합 app.py
- 한글 폰트(네모깨짐 방지) 자동 설정
- 초기금액 / 월 납입 / 배당 재투자(Adj Close) 옵션
- 확장 지표: 기간, CAGR, 연변동성, MDD, 최장 Underwater, Sharpe, Sortino, CAGR/최장UW, 시작/최종 잔고
- 결과 그래프: 누적가치, 언더워터, 포트폴리오 구성 파이차트
- 사이드바: 티커/비중, 날짜, 리밸런싱(연 1회/분기 1회/미적용), 수수료/슬리피지 (선택)

필수 설치: pip install streamlit yfinance pandas numpy matplotlib
런: streamlit run app.py
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# =====================================
# 0) 한글 폰트 설정 (네모깨짐 방지)
# =====================================
def set_korean_font():
    """가능한 경로에서 한글 폰트를 찾아 matplotlib에 등록.
    프로젝트 폴더에 ./fonts/NanumGothic.ttf 넣으면 가장 확실합니다.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf"),
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "NanumGothic.ttf"),
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    chosen = None
    for p in candidates:
        try:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                chosen = font_manager.FontProperties(fname=p).get_name()
                break
        except Exception:
            continue
    if chosen:
        rcParams["font.family"] = chosen
    else:
        rcParams["font.family"] = "NanumGothic, Apple SD Gothic Neo, Malgun Gothic, Noto Sans CJK KR, DejaVu Sans"
    rcParams["axes.unicode_minus"] = False

set_korean_font()

# =====================================
# 1) 기본 설정
# =====================================
st.set_page_config(page_title="ETF Backfill Portfolio Visualizer", layout="wide")
st.title("ETF Backfill Portfolio Visualizer")

# =====================================
# 2) 유틸 & 데이터 함수
# =====================================
@st.cache_data(show_spinner=False)
def yf_download(tickers: List[str], start: str, end: str, adj_close: bool=True) -> pd.DataFrame:
    """yfinance에서 가격 다운. Adj Close 사용 여부 선택."""
    cols = "Adj Close" if adj_close else "Close"
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        s = data[cols].copy()
    else:
        s = data.copy()
    s = s.dropna(how="all")
    s.index = pd.to_datetime(s.index)
    return s

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: float(w)/total for k, w in weights.items()}

def to_month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(idx).dt.to_period("M").dt.to_timestamp("M"))

def max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cummax = series.cummax()
    dd = series / cummax - 1.0
    mdd = dd.min()
    # 시작/최저 시점 찾기
    trough = dd.idxmin()
    peak = series.loc[:trough].idxmax()
    return float(mdd), peak, trough

def underwater_periods(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    uw = series/peak - 1.0
    return uw

def longest_underwater_months(series: pd.Series) -> int:
    uw = underwater_periods(series)
    below = (uw < 0).astype(int)
    # 연속 구간 길이
    max_len = 0
    cur = 0
    last = None
    for t, v in below.items():
        if v == 1:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    # uw의 빈도에 따라 월/일 변환
    if series.index.freqstr and series.index.freqstr.startswith("M"):
        return int(max_len)
    # 일반적으로 거래일 → 월로 환산
    return int(round(max_len/21))

def annualize_return(total_return: float, periods: int, per_year: float) -> float:
    if periods <= 0:
        return 0.0
    return (1 + total_return)**(per_year/periods) - 1

def annualized_vol(returns: pd.Series, per_year: float) -> float:
    return float(returns.std(ddof=0) * math.sqrt(per_year))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, per_year: float = 252.0) -> float:
    excess = returns - (rf/per_year)
    denom = returns.std(ddof=0)
    if denom == 0:
        return 0.0
    return float(excess.mean()/denom) * math.sqrt(per_year)

def sortino_ratio(returns: pd.Series, rf: float = 0.0, per_year: float = 252.0) -> float:
    excess = returns - (rf/per_year)
    downside = excess.copy()
    downside[downside > 0] = 0
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return float(excess.mean()/dd) * math.sqrt(per_year)

# =====================================
# 3) 사이드바 입력
# =====================================
with st.sidebar:
    st.header("옵션")
    start = st.date_input("시작일", value=datetime(2005,1,1)).strftime("%Y-%m-%d")
    end = st.date_input("종료일", value=datetime.today()).strftime("%Y-%m-%d")

    st.subheader("포트폴리오 구성(티커 / 비중%)")
    default_tickers = ["QQQ", "IEF", "TIP", "VCLT", "EMLC", "GDX", "MOO", "XLB", "VDE"]
    default_weights = [35, 20, 10, 10, 10, 7.5, 2.5, 2.5, 2.5]

    tickers: List[str] = []
    weights: List[float] = []
    cols = st.columns(3)
    n = st.number_input("항목 수", 1, 20, value=len(default_tickers), step=1)
    for i in range(n):
        t = st.text_input(f"티커 {i+1}", value=default_tickers[i] if i < len(default_tickers) else "")
        w = st.number_input(f"비중 {i+1}(%)", min_value=0.0, value=float(default_weights[i] if i < len(default_weights) else 0.0), step=1.0)
        tickers.append(t.strip().upper())
        weights.append(w)

    monthly_contrib = st.number_input("월 납입(원)", min_value=0, value=0, step=10000)
    initial_amount = st.number_input("초기 금액(원)", min_value=0, value=10000000, step=100000)
    reinvest_div = st.checkbox("배당 재투자(Adj Close 사용)", value=True)

    st.subheader("리밸런싱 & 비용")
    rebalance = st.selectbox("리밸런싱 주기", ["미적용", "연 1회", "분기 1회"]) 
    fee_bps = st.number_input("거래 수수료(bps)", min_value=0.0, value=0.0, step=1.0, help="한 번의 리밸런싱 체결 비용 가정")
    slippage_bps = st.number_input("슬리피지(bps)", min_value=0.0, value=0.0, step=1.0)

    run = st.button("백테스트 실행")

# 가중치 정규화
weight_map = normalize_weights({t: w for t, w in zip(tickers, weights) if t})

# =====================================
# 4) 백테스트 로직 (간단 버전)
# =====================================
@st.cache_data(show_spinner=True)
def backtest(tickers: List[str], weights: Dict[str, float], start: str, end: str, reinvest: bool,
             initial_amount: int, monthly_contrib: int, rebalance: str, fee_bps: float, slippage_bps: float):
    if not weights or sum(weights.values()) <= 0:
        return None

    # 가격 데이터
    price = yf_download(list(weights.keys()), start, end, adj_close=reinvest)
    price = price.dropna(how="all").ffill().dropna(how="any", axis=1)
    tickers_clean = [t for t in weights.keys() if t in price.columns]
    if not tickers_clean:
        return None
    w = normalize_weights({t: weights[t] for t in tickers_clean})

    # 일간 수익률
    ret = price.pct_change().fillna(0)

    # 리밸런싱 주기 정의
    if rebalance == "연 1회":
        rebal_dates = pd.date_range(start=ret.index.min(), end=ret.index.max(), freq="A")
    elif rebalance == "분기 1회":
        rebal_dates = pd.date_range(start=ret.index.min(), end=ret.index.max(), freq="Q")
    else:
        rebal_dates = pd.DatetimeIndex([])

    # 포트폴리오 시뮬레이션 (현금흐름 포함)
    # 초기
    value = initial_amount
    values = []
    alloc = {t: value * w[t] for t in w}

    last_rebal = ret.index.min()

    # 월납입 일정 (매월 말일에 납입 가정)
    month_ends = pd.date_range(start=ret.index.min(), end=ret.index.max(), freq="M")
    month_ends = set(pd.to_datetime(month_ends))

    for dt in ret.index:
        # 월 납입
        if monthly_contrib > 0 and dt in month_ends:
            value += monthly_contrib
            # 현재 비중대로 배분 매수 (간단 가정)
            for t in w:
                alloc[t] += monthly_contrib * w[t]

        # 일수익 반영
        for t in w:
            alloc[t] *= (1 + ret.loc[dt, t])

        # 리밸런싱
        if dt in rebal_dates:
            # 거래비용 반영 (간단히 총자산 * (fee+slip)/10000)
            total = sum(alloc.values())
            cost = total * ((fee_bps + slippage_bps)/10000.0)
            total_after = total - cost
            # 타깃 비중으로 재배분
            alloc = {t: total_after * w[t] for t in w}

        values.append((dt, sum(alloc.values())))

    curve = pd.Series({d:v for d, v in values}).sort_index()
    returns = curve.pct_change().fillna(0)

    # 지표 계산
    total_ret = curve.iloc[-1]/curve.iloc[0] - 1.0
    days = (curve.index[-1] - curve.index[0]).days
    years = days/365.25 if days>0 else 0.0
    cagr = (curve.iloc[-1]/curve.iloc[0])**(1/years) - 1 if years>0 else 0.0

    ann_vol = annualized_vol(returns, per_year=252.0)
    mdd, peak_dt, trough_dt = max_drawdown(curve)
    uw = underwater_periods(curve)
    longest_uw_m = longest_underwater_months(curve)

    shrp = sharpe_ratio(returns, rf=0.0, per_year=252.0)
    sortino = sortino_ratio(returns, rf=0.0, per_year=252.0)

    start_bal = float(curve.iloc[0])
    end_bal = float(curve.iloc[-1])

    results = {
        "curve": curve,
        "returns": returns,
        "uw": uw,
        "metrics": {
            "기간": f"{curve.index[0].date()} ~ {curve.index[-1].date()}",
            "CAGR": cagr,
            "연변동성": ann_vol,
            "최대낙폭(MDD)": mdd,
            "최장 Underwater(개월)": int(longest_uw_m),
            "Sharpe": shrp,
            "Sortino": sortino,
            "CAGR/최장UW": (cagr / longest_uw_m) if longest_uw_m>0 else 0.0,
            "Start Balance": start_bal,
            "End Balance": end_bal,
        },
        "weights": w,
    }
    return results

# =====================================
# 5) 시각화 헬퍼 (matplotlib로 한글 안전)
# =====================================
def plot_curve(series: pd.Series, title: str):
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("원")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_underwater(uw: pd.Series, title: str="Underwater(상고점 대비 수익률)"):
    fig, ax = plt.subplots()
    ax.fill_between(uw.index, uw.values, 0, step=None, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("비율")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_weights_pie(weight_map: Dict[str, float]):
    if not weight_map:
        st.info("구성이 비어 있습니다.")
        return
    labels = list(weight_map.keys())
    sizes = [weight_map[k]*100 for k in labels]
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.axis('equal')
    ax.set_title("포트폴리오 구성")
    st.pyplot(fig, clear_figure=True)

# =====================================
# 6) 실행 & 결과 표시
# =====================================
if run:
    res = backtest(tickers, weight_map, start, end, reinvest_div, initial_amount, monthly_contrib, rebalance, fee_bps, slippage_bps)
    if not res:
        st.error("데이터를 불러오지 못했습니다. 티커/기간/비중을 확인하세요.")
    else:
        curve = res["curve"]
        returns = res["returns"]
        uw = res["uw"]
        metrics = res["metrics"]

        # 상단 지표 카드
        st.subheader("핵심 지표")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("기간", metrics["기간"])
        k2.metric("CAGR", f"{metrics['CAGR']*100:,.2f}%")
        k3.metric("연변동성", f"{metrics['연변동성']*100:,.2f}%")
        k4.metric("최대낙폭", f"{metrics['최대낙폭(MDD)']*100:,.2f}%")
        k5.metric("최장 UW(개월)", f"{metrics['최장 Underwater(개월)']:,}")
        k6.metric("CAGR/최장UW", f"{metrics['CAGR/최장UW']:.4f}")

        k7, k8, k9 = st.columns(3)
        k7.metric("Sharpe", f"{metrics['Sharpe']:.3f}")
        k8.metric("Sortino", f"{metrics['Sortino']:.3f}")
        k9.metric("Start→End", f"{metrics['Start Balance']:,.0f} → {metrics['End Balance']:,.0f}")

        # 그래프
        st.subheader("결과 그래프")
        plot_curve(curve, "누적 포트폴리오 가치")
        plot_underwater(uw, "Underwater(고점대비) — 누적가치 기준")

        # 구성 파이 & 테이블
        st.subheader("포트폴리오 구성")
        c1, c2 = st.columns([1,1])
        with c1:
            plot_weights_pie(res["weights"])
        with c2:
            dfw = pd.DataFrame({"티커": list(res["weights"].keys()), "비중(%)": [v*100 for v in res["weights"].values()]})
            dfw = dfw.sort_values("비중(%)", ascending=False).reset_index(drop=True)
            st.dataframe(dfw, use_container_width=True)

else:
    st.info("사이드바에서 구성과 옵션을 선택하고 ‘백테스트 실행’을 눌러주세요.")
