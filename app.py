# ----------------------------------------------------------------------------
# ETF 백테스트 확장 분석기 (with Index Proxies)
# ----------------------------------------------------------------------------
import io
import math
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

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
    """Fetch Adj Close from Yahoo Finance (daily)."""
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
    proxy_overlap = proxy_px.loc[overlap_start:overlap_end]

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
        if (1.0 + r) != 0:
            synth.iloc[i] = synth.iloc[i + 1] / (1.0 + r)
        else:
            synth.iloc[i] = synth.iloc[i + 1]

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
    roll_max = s.cummax()
    drawdown = s / roll_max - 1
    mdd = drawdown.min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe (rf=0)": sharpe,
            "Max Drawdown": mdd, "Start": s.index[0].date(),
            "End": s.index[-1].date(), "Length (yrs)": years}


def fmt_pct(x):
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:,.2f}%"


# ------------------------------ UI ------------------------------
st.title("📈 ETF 백테스트 확장 분석기")
st.caption("ETF 상장 전 기간까지 추종지수로 백테스트하는 웹앱입니다.")

# ── 1) 포트폴리오 구성 ─────────────────────────────────────────────────────
st.sidebar.header("1) 포트폴리오 구성")

default_port = pd.DataFrame({
    "티커": ["QQQ", "IEF", "TIP", "VCLT", "EMLC", "GDX", "MOO", "XLB", "VDE"],
    "비율 (%)": [35.0, 20.0, 10.0, 10.0, 10.0, 7.5, 2.5, 2.5, 2.5],
})

if "portfolio_table" not in st.session_state:
    st.session_state["portfolio_table"] = default_port

edit_mode = st.sidebar.toggle("편집 모드", value=True, help="ON: 표를 직접 수정 / OFF: 합계 포함 읽기 전용 표")

if edit_mode:
    # 편집 가능한 표 (오직 이것만 렌더링)
    portfolio_df = st.sidebar.data_editor(
        st.session_state["portfolio_table"],
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor",
        column_config={
            "티커": st.column_config.TextColumn(
                "티커",
                help="예: QQQ, IEF, TIP",
                max_chars=15,
            ),
            "비율 (%)": st.column_config.NumberColumn(
                "비율 (%)",
                help="0~100 사이의 비율(%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f %%",
            ),
        },
    )
    # 입력 정리 & 세션 반영
    portfolio_df["티커"] = portfolio_df["티커"].astype(str).str.upper().str.strip()
    portfolio_df["비율 (%)"] = pd.to_numeric(portfolio_df["비율 (%)"], errors="coerce").fillna(0.0)
    st.session_state["portfolio_table"] = portfolio_df

else:
    # 합계 행 포함 '읽기 전용' 표 (오직 이것만 렌더링)
    portfolio_df = st.session_state["portfolio_table"].copy()
    portfolio_df["티커"] = portfolio_df["티커"].astype(str).str.upper().str.strip()
    portfolio_df["비율 (%)"] = pd.to_numeric(portfolio_df["비율 (%)"], errors="coerce").fillna(0.0)

    total_pct = float(portfolio_df["비율 (%)"].sum())
    display_df = portfolio_df.copy()
    sum_row = pd.DataFrame({"티커": ["합계"], "비율 (%)": [total_pct]})
    display_df = pd.concat([display_df, sum_row], ignore_index=True)

    def _style_totals(df: pd.DataFrame):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        last = df.index[-1]
        styles.loc[last, "티커"] = "font-weight: bold"
        if abs(df.loc[last, "비율 (%)"] - 100.0) < 1e-6:
            styles.loc[last, "비율 (%)"] = "font-weight: bold"
        else:
            styles.loc[last, "비율 (%)"] = "color: white; background-color: #d9534f; font-weight: bold"
        return styles

    st.sidebar.dataframe(
        display_df.style
            .format({"비율 (%)": "{:.1f}%"})
            .hide(axis="index")
            .apply(_style_totals, axis=None),
        use_container_width=True,
    )

# 자동 보정 버튼 (합을 100으로 정규화)
def normalize_weights():
    df = st.session_state["portfolio_table"].copy()
    s = pd.to_numeric(df["비율 (%)"], errors="coerce").fillna(0.0).sum()
    if s > 0:
        df["비율 (%)"] = pd.to_numeric(df["비율 (%)"], errors="coerce").fillna(0.0) * (100.0 / s)
        st.session_state["portfolio_table"] = df

st.sidebar.button("합계 100%로 자동 보정", on_click=normalize_weights)

# ── 2) 기간 설정 ────────────────────────────────────────────────────────────
st.sidebar.header("2) 기간 설정")
def_start = date(1980, 1, 1)
start_date = st.sidebar.date_input("시작일", def_start)
end_date = st.sidebar.date_input("종료일", date.today())

# ── 3) 추종지수 매핑 ────────────────────────────────────────────────────────
st.sidebar.header("3) 추종지수 매핑")
st.sidebar.caption("ETF와 그 추종지수(Proxy)를 설정합니다.")
proxy_map_default = pd.DataFrame({
    "ETF": ["QQQ", "IEF", "TIP", "VCLT", "EMLC", "GDX", "MOO", "XLB", "VDE"],
    "Proxy": ["^NDX", "^TNX", "^IRX", "^DJCB", "^EMHY", "^HUI", "^MXX", "^SP500-15", "^SP500-10"]
})
proxy_file = st.sidebar.file_uploader("CSV 업로드 (ETF,Proxy 형식)", type=["csv"])
proxy_df = proxy_map_default.copy()
if proxy_file is not None:
    try:
        proxy_df = pd.read_csv(proxy_file)
    except Exception as e:
        st.sidebar.error(f"CSV 읽기 실패: {e}")
proxy_df = st.sidebar.data_editor(proxy_df, num_rows="dynamic", use_container_width=True)

# ── 4) 옵션 ────────────────────────────────────────────────────────────────
st.sidebar.header("4) 옵션")
rebalance = st.sidebar.selectbox(
    "리밸런싱 주기",
    ["Monthly", "Quarterly", "Yearly"],
    index=0,
    help="포트폴리오를 재조정할 주기를 선택하세요."
)
log_scale = st.sidebar.checkbox("로그 스케일 차트", value=True)

# ── 5) 실행 ────────────────────────────────────────────────────────────────
st.sidebar.header("5) 실행")
run = st.sidebar.button("백테스트 실행", type="primary")

# ------------------------------ 실행 ------------------------------
if run:
    pf = st.session_state["portfolio_table"].copy()
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

    weights = {row["티커"]: row["비율 (%)"] / 100.0 for _, row in pf.iterrows()}

    mapping = {str(row.get("ETF", "")).upper(): str(row.get("Proxy", "")).upper()
               for _, row in proxy_df.iterrows() if str(row.get("ETF", "")).strip()}

    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

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
        st.dataframe(stats_tbl)

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
