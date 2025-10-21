# app.py

# ----------------------------------------------------------------------------

# ETF Backfill Portfolio Visualizer (with Index Proxies)

# Made beginner-friendly: just edit tickers/weights in the sidebar and press Run

# ----------------------------------------------------------------------------

import io

import math

import textwrap

from datetime import datetime, date

import numpy as np

import pandas as pd

import yfinance as yf

import streamlit as st

import matplotlib.pyplot as plt

# --- Korean font setup for matplotlib (ensure Hangul renders) ---

try:
    from matplotlib import font_manager, rcParams

    def _set_korean_font():
        candidates = [
            "AppleGothic",      # macOS default Korean
            "Malgun Gothic",    # Windows default Korean
            "NanumGothic",      # Common open font
            "Nanum Gothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        ]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                rcParams["font.family"] = name
                rcParams["axes.unicode_minus"] = False  # fix minus sign rendering
                return name
        # Fallback: at least fix minus sign even if Korean font not found
        rcParams["axes.unicode_minus"] = False
        return None

    _kr_font = _set_korean_font()

except Exception as e:
    # Safe fallback if font detection fails for any reason
    print(f"[Warning] Korean font setup skipped: {e}")
    pass
st.set_page_config(page_title="ETF Backfill Portfolio", layout="wide")

# ------------------------------

# Helper functions

# ------------------------------

def human_readable_list(items):

    items = [str(x) for x in items]

    if len(items) <= 2:

        return " and ".join(items)

    return ", ".join(items[:-1]) + f", and {items[-1]}"

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

def fetch_proxy_series(proxy_symbol: str, start: str, end: str) -> pd.Series:

    return fetch_prices_yf(proxy_symbol, start, end)

@st.cache_data(show_spinner=False)

def build_synthetic_from_proxy(etf: str, proxy: str, start: str, end: str) -> pd.Series:

    """Create extended price series using proxy returns before ETF inception.

    Logic:

    - Get daily price series for ETF and proxy over (start, end)

    - Find overlap window where both have data.

    - Compute daily returns for proxy in pre-overlap periods, then chain returns

      backwards from first ETF price to create synthetic history.

    - Post-overlap: use ETF actual prices.

    """

    etf_px = fetch_prices_yf(etf, start, end)

    proxy_px = fetch_proxy_series(proxy, start, end)

    if etf_px.empty and proxy_px.empty:

        return pd.Series(dtype=float)

    # If ETF has data from start already, just return it

    if not etf_px.empty and etf_px.index.min() <= pd.to_datetime(start):

        return etf_px

    # Determine overlap

    overlap_start = max(etf_px.index.min() if not etf_px.empty else pd.Timestamp.max,

                        proxy_px.index.min() if not proxy_px.empty else pd.Timestamp.max)

    overlap_end = min(etf_px.index.max() if not etf_px.empty else pd.Timestamp.min,

                      proxy_px.index.max() if not proxy_px.empty else pd.Timestamp.min)

    have_overlap = (overlap_start <= overlap_end)

    if not have_overlap:

        # No overlap — fall back: return proxy only, scaled to 100 at start

        if proxy_px.empty:

            return etf_px  # nothing we can do

        scaled = proxy_px / proxy_px.iloc[0] * 100.0

        scaled.name = etf

        return scaled

    # Compute returns

    etf_overlap = etf_px.loc[overlap_start:overlap_end]

    proxy_overlap = proxy_px.loc[overlap_start:overlap_end]

    # Use ETF first overlap price as anchor

    anchor_date = etf_overlap.index.min()

    anchor_price = float(etf_overlap.iloc[0])

    # Pre-overlap proxy section

    pre_proxy = proxy_px.loc[: anchor_date].iloc[:-1]  # strictly before anchor

    if pre_proxy.empty:

        # Nothing to extend, just return ETF prices

        return etf_px

    # Build backward synthetic using proxy returns

    proxy_ret = pre_proxy.pct_change().fillna(0.0)

    # We will reconstruct forward from earliest pre_proxy date up to anchor_date

    synth = pd.Series(index=pre_proxy.index.append(pd.Index([anchor_date])), dtype=float)

    synth.iloc[-1] = anchor_price

    # walk backwards from anchor_date to earliest pre_proxy date

    # price_{t-1} = price_t / (1 + r_t)

    for i in range(len(pre_proxy) - 1, -1, -1):

        r = proxy_ret.iloc[i]

        synth.iloc[i] = synth.iloc[i + 1] / (1.0 + r) if (1.0 + r) != 0 else synth.iloc[i + 1]

    # Combine synthetic (pre) + actual ETF (post)

    out = pd.concat([synth.iloc[:-1], etf_px.loc[anchor_date:]])

    out.name = etf

    return out

def monthly_rebalance(portfolio_df: pd.DataFrame, weights: dict) -> pd.Series:

    """Compute portfolio total return index with monthly rebalancing.

    portfolio_df: wide DataFrame of prices for tickers (Adj Close), daily frequency

    weights: dict of target weights that sum to 1

    Returns: portfolio value index (base 100)

    """

    prices = portfolio_df.dropna(how="all").copy()

    prices = prices.fillna(method="ffill").dropna()  # forward-fill gaps

    rets = prices.pct_change().dropna()

    # Monthly period boundaries

    month_ends = rets.resample('M').last().index

    tickers = list(weights.keys())

    w = pd.Series(weights)

    # Start with base value 100

    pv = pd.Series(index=rets.index, dtype=float)

    value = 100.0

    current_weights = w.copy()

    last_reb_date = None

    for dt, row in rets.iterrows():

        if (dt in month_ends) or (last_reb_date is None):

            # Rebalance at the start of period (use previous day value)

            current_weights = w.copy()

            last_reb_date = dt

        # Day's portfolio return = weighted sum of asset returns

        day_ret = (row[tickers] * current_weights[tickers]).sum()

        value *= (1.0 + day_ret)

        pv.loc[dt] = value

    return pv

def perf_stats(series: pd.Series) -> dict:

    s = series.dropna()

    if s.empty:

        return {}

    # Convert to daily returns

    rets = s.pct_change().dropna()

    if rets.empty:

        return {}

    # CAGR

    n_days = (s.index[-1] - s.index[0]).days

    years = n_days / 365.25 if n_days > 0 else 0

    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    # Volatility (daily -> annualized)

    vol = rets.std() * math.sqrt(252)

    # Sharpe (rf=0)

    sharpe = (rets.mean() * 252) / (rets.std() if rets.std() != 0 else np.nan)

    # Max Drawdown

    roll_max = s.cummax()

    drawdown = s / roll_max - 1

    mdd = drawdown.min()

    return {

        "CAGR": cagr,

        "Volatility": vol,

        "Sharpe (rf=0)": sharpe,

        "Max Drawdown": mdd,

        "Start": s.index[0].date(),

        "End": s.index[-1].date(),

        "Length (yrs)": years,

    }

def fmt_pct(x):

    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:,.2f}%"

# ------------------------------

# UI

# ------------------------------

st.title("📈 ETF Backfill Portfolio Visualizer테스트중")

st.caption("Analyze portfolios even before ETFs existed by splicing index proxies.")

st.sidebar.header("1) Portfolio Setup")

def_port = "QQQ:0.35, IEF:0.20, TIP:0.10, VCLT:0.10, EMLC:0.10, GDX:0.075, MOO:0.025, XLB:0.025, VDE:0.025"

raw = st.sidebar.text_input("Tickers and weights (ticker:weight, comma-separated)", def_port)

st.sidebar.header("2) Date Range")

def_start = date(1980, 1, 1)

start_date = st.sidebar.date_input("Start date", def_start)

end_date = st.sidebar.date_input("End date", date.today())

st.sidebar.header("3) Proxy Mapping")

st.sidebar.caption("Edit if you like. Left column = ETF, Right column = Proxy index/ticker.")

# Default mapping for common ETFs in the user's KB_Portfolio

proxy_map_default = pd.DataFrame(

    {

        "ETF": [

            "QQQ", "IEF", "TIP", "VCLT", "EMLC", "GDX", "MOO", "XLB", "VDE",

        ],

        "Proxy": [

            "^NDX",               # NASDAQ-100 index

            "^TNX",               # 10Y Treasury Yield (not a price index, but usable to simulate)

            "^IRX",               # 13-week T-Bill rate as rough placeholder (user can change)

            "^DJCB",              # Dow Jones Corporate Bond Index (approx)

            "^EMHY",              # Emerging markets HY proxy (approx)

            "^HUI",               # Gold BUGS Index

            "^MXX",               # Mexico IPC as rough agri industry proxy (placeholder)

            "^SP500-15",          # S&P 500 Materials sector (Yahoo sometimes: ^SP500-40). Edit if missing.

            "^SP500-10",          # S&P 500 Energy sector (Yahoo sometimes: ^SP500-10). Edit if missing.

        ],

    }

)

st.sidebar.write("You can also upload a CSV with columns 'ETF,Proxy' to override:")

proxy_file = st.sidebar.file_uploader("Proxy mapping CSV", type=["csv"])  # optional

proxy_df = proxy_map_default.copy()

if proxy_file is not None:

    try:

        proxy_df = pd.read_csv(proxy_file)

    except Exception as e:

        st.sidebar.error(f"Failed to read CSV: {e}")

proxy_df = st.sidebar.data_editor(proxy_df, num_rows="dynamic", use_container_width=True)

st.sidebar.header("4) Options")
rebalance = st.sidebar.selectbox(
    "Rebalance frequency",
    ["Monthly", "Quarterly", "Yearly"],
    index=0,
    help="Select how often to rebalance the portfolio"
)
log_scale = st.sidebar.checkbox("Log scale charts", value=True)
st.sidebar.header("5) Run")

run = st.sidebar.button("Run Backtest", type="primary")

st.markdown(

    """

    **How this works**  

    If an ETF has limited history, we fetch a proxy index with longer history. We splice

    proxy *returns* before the ETF's inception, anchoring at the first overlap date, to create

    a synthetic pre-inception series. Then we backtest your portfolio on the extended history.

    """

)

# Parse portfolio input

weights = {}

try:

    for part in raw.split(","):

        if not part.strip():

            continue

        t, w = part.split(":")

        weights[t.strip().upper()] = float(w.strip())

except Exception:

    st.error("Please enter tickers and weights in the format: TICKER:weight, TICKER:weight ...")

# Normalize weights

if weights:

    s = sum(weights.values())

    if s <= 0:

        st.error("Weights must sum to a positive number.")

    else:

        weights = {k: v / s for k, v in weights.items()}

# Build mapping dict

mapping = {row["ETF"].upper(): str(row["Proxy"]).upper() for _, row in proxy_df.iterrows() if str(row.get("ETF", "")).strip()}

if run and weights:

    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")

    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    tabs = st.tabs(["Portfolio", "Constituents", "Settings & Notes"])

    # Build extended series for each ticker

    all_prices = {}

    notes = []

    for t in weights.keys():

        # Fetch ETF

        etf_px = fetch_prices_yf(t, start, end)

        if t in mapping and mapping[t]:

            proxy = mapping[t]

        else:

            proxy = None

        if proxy:

            synth = build_synthetic_from_proxy(t, proxy, start, end)

            if synth.empty:

                notes.append(f"{t}: no data found for ETF nor proxy {proxy}.")

            else:

                all_prices[t] = synth

                if not etf_px.empty and synth.index.min() < etf_px.index.min():

                    notes.append(f"{t}: extended with proxy {proxy} back to {synth.index.min().date()} (ETF starts {etf_px.index.min().date()}).")

                else:

                    notes.append(f"{t}: used available ETF history only (no extension applied).")

        else:

            if etf_px.empty:

                notes.append(f"{t}: no data found and no proxy provided.")

            else:

                all_prices[t] = etf_px

                notes.append(f"{t}: used available ETF history only (no proxy).")

    if not all_prices:

        st.error("No price data available — check tickers and proxies.")

    else:

        price_df = pd.concat(all_prices.values(), axis=1)

        price_df = price_df.sort_index().ffill().dropna()

        # Portfolio series

        if rebalance == "Monthly":

            pv = monthly_rebalance(price_df, weights)

        else:

            pv = monthly_rebalance(price_df, weights)

        idx = pv / pv.iloc[0] * 100.0

        with tabs[0]:

            st.subheader("Portfolio Equity Curve")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(idx.index, idx.values, label="Portfolio (base=100)")

            ax.set_ylabel("Index (base=100)")

            if log_scale:

                ax.set_yscale('log')

            ax.grid(True, linestyle='--', alpha=0.4)

            ax.legend()

            st.pyplot(fig)

            # Drawdown

            st.subheader("Drawdown")

            rolling_max = idx.cummax()

            dd = idx / rolling_max - 1

            fig2, ax2 = plt.subplots(figsize=(10, 3))

            ax2.fill_between(dd.index, dd.values, 0.0)

            ax2.set_ylabel("Drawdown")

            ax2.grid(True, linestyle='--', alpha=0.4)

            st.pyplot(fig2)

            # Stats table

            st.subheader("Performance Stats")

            stats = perf_stats(idx)

            stats_tbl = pd.DataFrame(

                {

                    "Value": [

                        fmt_pct(stats.get("CAGR")),

                        fmt_pct(stats.get("Volatility")),

                        fmt_pct(stats.get("Sharpe (rf=0)")),

                        fmt_pct(stats.get("Max Drawdown")),

                        stats.get("Start"),

                        stats.get("End"),

                        f"{stats.get('Length (yrs)', 0):.1f}",

                    ]

                },

                index=["CAGR", "Volatility (ann.)", "Sharpe (rf=0)", "Max Drawdown", "Start", "End", "Length (yrs)"]

            )

            st.dataframe(stats_tbl)

        with tabs[1]:

            st.subheader("Constituent Price Series (base=100)")

            base = price_df / price_df.iloc[0] * 100.0

            fig3, ax3 = plt.subplots(figsize=(10, 5))

            for c in base.columns:

                ax3.plot(base.index, base[c].values, label=c)

            ax3.set_ylabel("Index (base=100)")

            if log_scale:

                ax3.set_yscale('log')

            ax3.grid(True, linestyle='--', alpha=0.4)

            ax3.legend(ncols=3)

            st.pyplot(fig3)

            st.caption("Tip: If a line ends early, add or correct a proxy mapping in the sidebar and rerun.")

        with tabs[2]:

            st.subheader("Settings & Notes")

            st.markdown("- Rebalance: monthly.\n- Data source: Yahoo Finance via yfinance (auto-adjusted closes).\n- Proxy extension: splices **proxy returns** before the ETF inception anchored at first overlap date.\n- You can upload your own ETF→Proxy CSV mapping in the sidebar.")

            if notes:

                st.write("**Run notes:**")

                for n in notes:

                    st.write("- ", n)

    # Download data

    st.divider()

    st.subheader("Download Data")

    if 'price_df' in locals():

        buf = io.BytesIO()

        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:

            price_df.to_excel(writer, sheet_name='Prices')

            if 'idx' in locals():

                idx.to_frame('Portfolio_Index').to_excel(writer, sheet_name='Portfolio')

        st.download_button(

            label="Download prices & portfolio (Excel)",

            data=buf.getvalue(),

            file_name="portfolio_backfill_data.xlsx",

            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

        )

# Footer

st.caption(

    "Note: Some default proxies are placeholders. Edit them to better-matched indexes (e.g., Bloomberg Barclays for TIP/VCLT, MSCI sector indices, NASDAQ-100 for QQQ). If Yahoo symbol missing, replace with an available one.")


