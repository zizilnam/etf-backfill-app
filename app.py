# ----------------------------------------------------------------------------
# ETF 백테스트 확장 분석기 (with Index Proxies) — Auto Max Period
# ----------------------------------------------------------------------------
import io
import math
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# --- 대표 포트폴리오 비교 섹션에 '설명'을 붙이는 드롭인 스니펫 ---
# 이 블록만 복사해서 app.py의 "대표 포트폴리오 비교" 자리에 붙여넣으면 됩니다.
# 핵심: PRESETS 딕셔너리에서 desc/why 필드로 설명을 정의하고, 렌더러가 이를 보여줍니다.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1) 데이터 정의: 구성 + 설명 (원하면 티커/비중 자유 수정)
PRESETS_WITH_DESC = {
    "60:40 포트폴리오": {
        "desc": "성장(주식)과 안정(채권)을 단순하게 결합한 기본형. 시장 평균에 가까운 성과를 목표로 합니다.",
        "why": [
            "주식의 장기 성장 + 채권의 완충 효과로 변동성을 낮춤",
            "리밸런싱(정기 비중 복원)으로 위험 관리가 쉬움",
            "초보자도 이해하기 쉬운 구조"
        ],
        "composition": [
            {"티커": "SPY", "자산": "미국 주식", "비중(%)": 60},
            {"티커": "BND", "자산": "미국 종합채권", "비중(%)": 40},
        ],
    },
    "올웨더 포트폴리오": {
        "desc": "경제의 네 가지 국면(성장/침체/인플레/디플레)에서 모두 버티도록 설계된 레이 달리오식 자산 배분.",
        "why": [
            "장·중기 국채 비중으로 경기 둔화·디플레 리스크 대응",
            "금·원자재로 인플레이션 환경 방어",
            "주식 비중은 낮추되 전체 포트 변동성 균형 지향"
        ],
        "composition": [
            {"티커": "VTI",  "자산": "미국 주식",       "비중(%)": 30},
            {"티커": "VGLT", "자산": "미국 장기국채",   "비중(%)": 40},
            {"티커": "IEF",  "자산": "미국 중기국채",   "비중(%)": 15},
            {"티커": "IAU",  "자산": "금",           "비중(%)": 7.5},
            {"티커": "DBC",  "자산": "원자재",       "비중(%)": 7.5},
        ],
    },
    "GAA 포트폴리오": {
        "desc": "멥 페이버 계열의 글로벌 자산 배분 예시. 주식·채권·리츠·원자재·금·현금 등 광범위한 분산.",
        "why": [
            "전 세계 자산군으로 폭넓게 분산",
            "리츠·원자재·금 포함으로 실물·인플레 헷지 강화",
            "현금/단기채 비중으로 유동성도 확보"
        ],
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

# (선택) 임시 성과값 – 실제 백테스트 결과와 연결해 교체하세요
METRICS = {
    "60:40 포트폴리오": {"연평균 수익률": "7.2%", "변동성": "9.5%", "최대 낙폭": "-25%"},
    "올웨더 포트폴리오": {"연평균 수익률": "6.8%", "변동성": "6.0%", "최대 낙폭": "-15%"},
    "GAA 포트폴리오": {"연평균 수익률": "7.5%", "변동성": "8.0%", "최대 낙폭": "-20%"},
}

# 2) 렌더 함수: 설명 + 표 + 파이차트 + 성과

def render_rep_portfolios_with_desc():
    st.markdown("### 🚀 대표 포트폴리오 비교")
    for name, spec in PRESETS_WITH_DESC.items():
        st.markdown(f"#### 📊 {name}")
        st.caption(spec["desc"])  # <-- 설명 한 줄

        # (선택) 왜 이런 비율인가? 자세한 설명은 접었다 펴는 형식
        with st.expander("📘 왜 이런 비율인가?", expanded=False):
            for bullet in spec.get("why", []):
                st.markdown(f"- {bullet}")

        # 레이아웃: 좌(구성표/성과), 우(파이그래프)
        left, right = st.columns([1.2, 1])
        df = pd.DataFrame(spec["composition"])  # 티커/자산/비중

        with left:
            st.dataframe(df, hide_index=True, use_container_width=True)
            m = METRICS.get(name)
            if m:
                st.write(
                    f"- **연평균 수익률:** {m['연평균 수익률']}  \n"
                    f"- **변동성:** {m['변동성']}  \n"
                    f"- **최대 낙폭:** {m['최대 낙폭']}"
                )

        with right:
            _pie(df)

        st.markdown("---")


def _pie(df: pd.DataFrame):
    sizes = df["비중(%)"].astype(float).tolist()
    labels = (df["자산"] + " (" + df["비중(%)"].astype(str) + "%)").tolist()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# 3) 사용 방법 (한 줄): 원하는 위치에서 아래 함수를 호출하세요.
# render_rep_portfolios_with_desc()

# --- 한국어 폰트 설정 (matplotlib 한글 깨짐 방지) ---
try:
    from matplotlib import font_manager, rcParams

    def _set_korean_font():
        candidates = [
            "AppleGothic", "Malgun Gothic", "NanumGothic", "Nanum Gothic",
            "Noto Sans CJK KR", "Noto Sans KR"
        ]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                rcParams["font.family"] = name
                rcParams["axes.unicode_minus"] = False
                return name
        rcParams["axes.unicode_minus"] = False
        return None

    _set_korean_font()
except Exception as e:
    print(f"[Warning] Korean font setup skipped: {e}")
    pass

st.set_page_config(page_title="ETF 백테스트 확장 분석기", layout="wide")

# ------------------------------ Helper functions ------------------------------
@st.cache_data(show_spinner=False)
def fetch_prices_yf(symbol: str, start: str, end: str) -> pd.Series:
    """Fetch Close (auto_adjusted) from Yahoo Finance (daily)."""
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    s = data["Close"].copy()
    s.name = symbol
    return s


@st.cache_data(show_spinner=False)
def build_synthetic_from_proxy(etf: str, proxy: str, start: str, end: str) -> pd.Series:
    """Extend ETF price history using proxy returns before inception."""
    etf_px = fetch_prices_yf(etf, start, end)
    proxy_px = fetch_prices_yf(proxy, start, end)

    if etf_px.empty and proxy_px.empty:
        return pd.Series(dtype=float)
    if not etf_px.empty and etf_px.index.min() <= pd.to_datetime(start):
        return etf_px

    overlap_start = max(etf_px.index.min() if not etf_px.empty else pd.Timestamp.max,
                        proxy_px.index.min() if not proxy_px.empty else pd.Timestamp.max)
    overlap_end = min(etf_px.index.max() if not etf_px.empty else pd.Timestamp.min,
                      proxy_px.index.max() if not proxy_px.empty else pd.Timestamp.min)
    have_overlap = (overlap_start <= overlap_end)

    if not have_overlap:
        if proxy_px.empty:
            return etf_px
        scaled = proxy_px / proxy_px.iloc[0] * 100.0
        scaled.name = etf
        return scaled

    etf_overlap = etf_px.loc[overlap_start:overlap_end]
    anchor_date = etf_overlap.index.min()
    anchor_price = float(etf_overlap.iloc[0])
    pre_proxy = proxy_px.loc[: anchor_date].iloc[:-1]
    if pre_proxy.empty:
        return etf_px

    proxy_ret = pre_proxy.pct_change().fillna(0.0)
    synth = pd.Series(index=pre_proxy.index.append(pd.Index([anchor_date])), dtype=float)
    synth.iloc[-1] = anchor_price
    for i in range(len(pre_proxy) - 1, -1, -1):
        r = float(proxy_ret.iloc[i])
        synth.iloc[i] = synth.iloc[i + 1] / (1.0 + r) if (1.0 + r) != 0 else synth.iloc[i + 1]

    out = pd.concat([synth.iloc[:-1], etf_px.loc[anchor_date:]])
    out.name = etf
    return out


def flexible_rebalance(portfolio_df: pd.DataFrame, weights: dict, freq: str = "Monthly") -> pd.Series:
    """Adjustable rebalancing: Monthly / Quarterly / Yearly"""
    prices = portfolio_df.dropna(how="all").fillna(method="ffill").dropna()
    rets = prices.pct_change().dropna()

    if freq == "Monthly":
        rebalance_dates = rets.resample("M").last().index
    elif freq == "Quarterly":
        rebalance_dates = rets.resample("Q").last().index
    elif freq == "Yearly":
        rebalance_dates = rets.resample("Y").last().index
    else:
        rebalance_dates = rets.resample("M").last().index

    tickers = list(weights.keys())
    w = pd.Series(weights)
    pv = pd.Series(index=rets.index, dtype=float)
    value = 100.0
    current_weights = w.copy()
    last_reb_date = None

    for dt, row in rets.iterrows():
        if (dt in rebalance_dates) or (last_reb_date is None):
            current_weights = w.copy()
            last_reb_date = dt
        day_ret = (row[tickers] * current_weights[tickers]).sum()
        value *= (1.0 + day_ret)
        pv.loc[dt] = value
    return pv


def perf_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    rets = s.pct_change().dropna()
    if rets.empty:
        return {}
    n_days = (s.index[-1] - s.index[0]).days
    years = n_days / 365.25 if n_days > 0 else 0
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * math.sqrt(252)
    sharpe = (rets.mean() * 252) / (rets.std() if rets.std() != 0 else np.nan)
    mdd = (s / s.cummax() - 1).min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe (rf=0)": sharpe,
            "Max Drawdown": mdd, "Start": s.index[0].date(),
            "End": s.index[-1].date(), "Length (yrs)": (n_days / 365.25)}


def fmt_pct(x):
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:,.2f}%"


# ------------------------------ UI ------------------------------
st.title("📈 ETF 백테스트 확장 분석기")
import streamlit as st
import matplotlib.pyplot as plt

# --- 사용 가이드 ---
st.markdown("### 🧭 환영합니다!")
st.caption("ETF 상장 전 기간까지 추종지수로 백테스트할 수 있는 투자 입문자용 웹앱입니다.")

st.write("""
이 웹앱은 **ETF 상장 이전 기간**까지도 추종지수를 이용해 백테스트할 수 있어, 
과거부터 지금까지의 장기 성과를 살펴볼 수 있습니다.

처음 오신 분들도 쉽게 이해할 수 있도록, 포트폴리오의 기본 원리와 대표 예시를 함께 제공합니다.
""")


# --- 기존 문구 대신 친절한 인트로 ---
st.markdown("---")
st.markdown("### 🧩 사용 방법")
st.write("""
1️⃣ 분석하고 싶은 ETF를 선택합니다.  
2️⃣ 각 자산의 비중을 입력합니다.  
3️⃣ 기간을 설정하면 → 자동으로 **상장 전 추종지수 포함 백테스트 결과**를 볼 수 있습니다!
""")

# --- 포트폴리오 이론 요약 ---
st.markdown("---")
st.markdown("### 💡 포트폴리오란?")
st.write("""
**포트폴리오**는 여러 자산(예: 주식, 채권, 금 등)에 분산 투자하여 
한쪽 자산의 손실을 다른 자산의 수익으로 상쇄하도록 설계된 투자 조합입니다.
""")

# --- 대표 포트폴리오 비교 ---
st.markdown("---")

render_rep_portfolios_with_desc()



# ── 1) 포트폴리오 구성 (항상 '합계' 포함 단일 표) + 자동 추종지수 & 프록시 매핑 ─────────
st.sidebar.header("1) 포트폴리오 구성")

# ── 간단 티커 → 추종지수/프록시 매핑 테이블 (필요 시 계속 보강) ──
_AUTO_MAP = {
    "QQQ":  {"Index": "NASDAQ-100", "Provider": "Nasdaq", "ProxyTicker": "^NDX", "Notes": ""},
    "IEF":  {"Index": "ICE U.S. Treasury 7–10 Year", "Provider": "ICE", "ProxyTicker": "", "Notes": ""},
    "TIP":  {"Index": "Bloomberg U.S. TIPS", "Provider": "Bloomberg", "ProxyTicker": "", "Notes": ""},
    "VCLT": {"Index": "Bloomberg U.S. Long Corporate", "Provider": "Bloomberg", "ProxyTicker": "", "Notes": ""},
    "EMLC": {"Index": "JPM GBI-EM GD (LC)", "Provider": "JPMorgan", "ProxyTicker": "", "Notes": ""},
    "GDX":  {"Index": "NYSE Arca Gold Miners", "Provider": "NYSE Arca", "ProxyTicker": "", "Notes": ""},
    "MOO":  {"Index": "MVIS Global Agribusiness", "Provider": "MV Index", "ProxyTicker": "", "Notes": ""},
    "XLB":  {"Index": "S&P Materials Select Sector", "Provider": "S&P DJI", "ProxyTicker": "", "Notes": ""},
    "VDE":  {"Index": "MSCI US IMI Energy 25/50", "Provider": "MSCI", "ProxyTicker": "", "Notes": ""},
    # "SPY": {"Index": "S&P 500", "Provider": "S&P DJI", "ProxyTicker": "^GSPC", "Notes": ""},
}

def _auto_index_meta(ticker: str) -> dict:
    t = (ticker or "").strip().upper()
    return _AUTO_MAP.get(t, {"Index": "알 수 없음", "Provider": "", "ProxyTicker": "", "Notes": ""})

def _auto_index_label(ticker: str) -> str:
    meta = _auto_index_meta(ticker)
    base = meta.get("Index", "")
    provider = meta.get("Provider", "")
    proxy = meta.get("ProxyTicker", "")
    label = f"{base} ({provider})" if base else "알 수 없음"
    if proxy:
        label += f" / proxy: {proxy}"
    return label or "알 수 없음"

# ── 기본 포트폴리오 ──
default_port = pd.DataFrame({
    "티커": ["QQQ", "IEF", "TIP", "VCLT", "EMLC", "IAU", "BCI"],
    "비율 (%)": [35.0, 20.0, 10.0, 10.0, 10.0, 7.5, 7.5],
})

# 세션 초기화
if "portfolio_rows" not in st.session_state:
    st.session_state["portfolio_rows"] = default_port

def _append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["티커"] = base["티커"].astype(str).str.upper().str.strip()
    base["비율 (%)"] = pd.to_numeric(base["비율 (%)"], errors="coerce").fillna(0.0)
    total = float(base["비율 (%)"].sum())
    total_row = pd.DataFrame({"티커": ["합계"], "비율 (%)": [total]})
    return pd.concat([base, total_row], ignore_index=True)

def _attach_auto_index(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["추종지수(자동)"] = out["티커"].apply(
        lambda x: "—" if str(x).strip().upper() == "합계" else _auto_index_label(str(x))
    )
    return out

# 편집 가능한 단일 표(합계 포함 / 3열 자동)
_editor_df = _append_total_row(st.session_state["portfolio_rows"])
_editor_df = _attach_auto_index(_editor_df)

edited_df_out = st.sidebar.data_editor(
    _editor_df,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "티커": st.column_config.TextColumn("티커", help="예: QQQ, IEF, TIP", max_chars=15),
        "비율 (%)": st.column_config.NumberColumn(
            "비율 (%)", help="0~100 사이의 비율(%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f %%"
        ),
        "추종지수(자동)": st.column_config.TextColumn("추종지수(자동)", help="티커 기반 자동 매핑", disabled=True),
    },
    disabled=["추종지수(자동)"],  # 3열은 읽기 전용
)

def _sanitize_user_edit(df_with_total_and_auto: pd.DataFrame) -> pd.DataFrame:
    df = df_with_total_and_auto.copy()
    # 계산 컬럼 제거
    if "추종지수(자동)" in df.columns:
        df = df.drop(columns=["추종지수(자동)"])
    # '합계' 행 제거
    df = df[df["티커"].astype(str).str.strip().str.upper() != "합계"]
    # 공백/0 정리
    df["티커"] = df["티커"].astype(str).str.upper().str.strip()
    df["비율 (%)"] = pd.to_numeric(df["비율 (%)"], errors="coerce").fillna(0.0)
    df = df[df["티커"].str.len() > 0]
    return df

# 세션 반영
st.session_state["portfolio_rows"] = _sanitize_user_edit(edited_df_out)

# 합계 경고(표 아래 한 줄)
_current_total = float(st.session_state["portfolio_rows"]["비율 (%)"].sum())
if abs(_current_total - 100.0) < 1e-6:
    st.sidebar.caption(f"✅ 합계: **{_current_total:.1f}%**")
elif _current_total < 100.0:
    st.sidebar.caption(f":red[⚠ 합계 {_current_total:.1f}% — 100% 미만]")
else:
    st.sidebar.caption(f":red[⚠ 합계 {_current_total:.1f}% — 100% 초과]")

# 자동 보정 버튼
def _normalize_weights():
    df = st.session_state["portfolio_rows"].copy()
    s = pd.to_numeric(df["비율 (%)"], errors="coerce").fillna(0.0).sum()
    if s > 0:
        df["비율 (%)"] = pd.to_numeric(df["비율 (%)"], errors="coerce").fillna(0.0) * (100.0 / s)
        st.session_state["portfolio_rows"] = df

st.sidebar.button("합계 100%로 자동 보정", on_click=_normalize_weights)

# ── 수동 매핑 없이 proxy_df/mapping 자동 생성 ──
def _build_auto_proxy_df(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    tickers = (
        portfolio_df["티커"]
        .astype(str).str.upper().str.strip()
    )
    tickers = [t for t in tickers.unique() if t and t != "합계"]
    rows = []
    for t in tickers:
        meta = _auto_index_meta(t)
        rows.append({
            "ETF": t,
            "Index": meta.get("Index", ""),
            "Provider": meta.get("Provider", ""),
            "Proxy": meta.get("ProxyTicker", "") or "",  # 하위 코드에서 'Proxy' 컬럼 사용
            "Notes": meta.get("Notes", "") or "",
        })
    return pd.DataFrame(rows, columns=["ETF", "Index", "Provider", "Proxy", "Notes"])

# 항상 proxy_df와 mapping이 존재하도록 생성
proxy_df = _build_auto_proxy_df(st.session_state["portfolio_rows"])
mapping = {
    str(row.get("ETF", "")).upper(): str(row.get("Proxy", "")).upper()
    for _, row in proxy_df.iterrows() if str(row.get("ETF", "")).strip()
}

# ── 2) 옵션 ─────────────────────────────────────────────────────
st.sidebar.header("2) 옵션")
rebalance = st.sidebar.selectbox(
    "리밸런싱 주기", ["Monthly", "Quarterly", "Yearly"], index=0,
    help="포트폴리오를 재조정할 주기를 선택하세요."
)
log_scale = st.sidebar.checkbox("로그 스케일 차트", value=True)

# ── 3) 실행 ─────────────────────────────────────────────────────
st.sidebar.header("3) 실행")
run = st.sidebar.button("백테스트 실행", type="primary")

# ------------------------------ 실행 ------------------------------
if run:
    pf = st.session_state["portfolio_rows"].copy()
    pf["티커"] = pf["티커"].astype(str).str.upper().str.strip()
    pf["비율 (%)"] = pd.to_numeric(pf["비율 (%)"], errors="coerce").fillna(0.0)
    pf = pf[(pf["티커"].str.len() > 0) & (pf["비율 (%)"] > 0)]

    total_pct_now = float(pf["비율 (%)"].sum())
    if total_pct_now <= 0:
        st.error("비율 합은 0보다 커야 합니다.")
        st.stop()
    if abs(total_pct_now - 100.0) > 1e-6:
        st.error(f"포트폴리오 비율 총합이 {total_pct_now:.1f}% 입니다. 합계가 정확히 100%가 되도록 조정하세요.")
        st.info("TIP: 사이드바의 ‘합계 100%로 자동 보정’ 버튼을 눌러 즉시 맞출 수 있습니다.")
        st.stop()

    # 가중치 dict (0~1)
    weights = {row["티커"]: row["비율 (%)"] / 100.0 for _, row in pf.iterrows()}

    # ▶ 기간 자동: 가능한 최장 기간
    start = "1900-01-01"
    end = pd.to_datetime(date.today()).strftime("%Y-%m-%d")

    tabs = st.tabs(["포트폴리오", "구성종목", "설정 및 참고"])
    all_prices, notes = {}, []

    for t in weights.keys():
        etf_px = fetch_prices_yf(t, start, end)
        proxy = mapping.get(t)
        if proxy:
            synth = build_synthetic_from_proxy(t, proxy, start, end)
            if synth.empty:
                notes.append(f"{t}: 데이터 없음 ({proxy} 불러오기 실패)")
            else:
                all_prices[t] = synth
                notes.append(f"{t}: {proxy} 지수로 연장 ({synth.index.min().date()}까지)")
        else:
            if etf_px.empty:
                notes.append(f"{t}: 데이터 없음")
            else:
                all_prices[t] = etf_px
                notes.append(f"{t}: ETF 데이터만 사용")

    if not all_prices:
        st.error("유효한 데이터가 없습니다. 티커나 프록시를 확인하세요.")
        st.stop()

    price_df = pd.concat(all_prices.values(), axis=1).sort_index().ffill().dropna()
    pv = flexible_rebalance(price_df, weights, freq=rebalance)
    idx = pv / pv.iloc[0] * 100.0

    with tabs[0]:
        st.subheader("📊 포트폴리오 지수 추이")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(idx.index, idx.values, label="Portfolio (base=100)")
        ax.set_ylabel("지수 (기준=100)")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 드로우다운(누적 손실률)")
        rolling_max = idx.cummax()
        dd = idx / rolling_max - 1
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.fill_between(dd.index, dd.values, 0.0)
        ax2.set_ylabel("손실률")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)

        st.subheader("📈 성과 지표")
        stats = perf_stats(idx)
        stats_tbl = pd.DataFrame({
            "Value": [
                fmt_pct(stats.get("CAGR")),
                fmt_pct(stats.get("Volatility")),
                fmt_pct(stats.get("Sharpe (rf=0)")),
                fmt_pct(stats.get("Max Drawdown")),
                stats.get("Start"),
                stats.get("End"),
                f"{stats.get('Length (yrs)', 0):.1f}",
            ]
        }, index=["CAGR", "Volatility (연율)", "Sharpe (rf=0)", "최대손실", "시작", "종료", "기간(년)"])
        st.dataframe(stats_tbl, use_container_width=True)

    with tabs[1]:
        st.subheader("📊 개별 ETF 가격 추이 (기준=100)")
        base = price_df / price_df.iloc[0] * 100.0
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for c in base.columns:
            ax3.plot(base.index, base[c].values, label=c)
        ax3.set_ylabel("지수 (기준=100)")
        if log_scale:
            ax3.set_yscale("log")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(ncols=3)
        st.pyplot(fig3)
        st.caption("라인이 일찍 끝나면 프록시 매핑을 확인하세요.")

    with tabs[2]:
        st.subheader("⚙️ 설정 및 참고")
        st.markdown(
            f"- 리밸런싱: **{rebalance}**\n"
            "- 데이터 출처: Yahoo Finance (yfinance)\n"
            f"- 기간: 자동 최대 (start={start}, end={end})\n"
            "- 프록시 확장: ETF 상장 전 구간을 추종지수 수익률로 보완\n"
            "- CSV 업로드 기능으로 직접 매핑 가능"
        )
        if notes:
            st.write("**실행 노트:**")
            for n in notes:
                st.write("- ", n)

    st.divider()
    st.subheader("📂 데이터 다운로드")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        price_df.to_excel(writer, sheet_name="Prices")
        idx.to_frame("Portfolio_Index").to_excel(writer, sheet_name="Portfolio")
    st.download_button(
        label="엑셀 파일 다운로드",
        data=buf.getvalue(),
        file_name="portfolio_backfill_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("⚠️ 일부 프록시는 대체용 심볼입니다. 필요시 직접 교체하세요.")






