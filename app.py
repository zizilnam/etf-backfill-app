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
    retu
