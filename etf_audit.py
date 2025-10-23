"""
모듈 목적
- 전체 ETF 목록을 점검해, '추종지수 미확인' 항목을 자동으로 찾아내고, 
  규칙 기반(키워드/자산군)으로 합리적인 프록시 지수를 자동 매핑합니다.
- IAU/BCI 같은 실물/원자재형 ETF도 자동으로 백테스트 가능하게 만듭니다.

사용법(요약)
1) 아래 파일을 `etf_audit.py`로 저장하거나 app.py에 포함하세요.
2) 앱/백엔드에서 `audit_and_autofix_proxies(etf_tickers)`를 호출하면
   - 누락 프록시 감지 → 규칙 기반 프록시 제안/적용 → 리포트 DataFrame 반환
3) 반환된 리포트를 화면에 표시하고, 최종 `PROXY_MAP`을 JSON으로 저장하여 재사용하세요.

필수 라이브러리: yfinance, pandas, numpy, pandas_datareader(FRED 사용 시)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------
# (선택) Streamlit에서 사용할 때 캐시 데코레이터
# -------------------------------------------------------
try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(func):
        return func

# -------------------------------------------------------
# 기존 프록시 매핑과 하이브리드 함수(앞서 제공한 것과 호환)
# -------------------------------------------------------
@dataclass
class ProxySpec:
    source: str      # "FRED" | "YF" | "CUSTOM"
    series: str      # 예: FRED 코드, Yahoo 심볼, 커스텀 식별자
    name: str        # 설명용
    transform: str = "identity"  # 필요 시 변환

# 기존/기본 프록시 매핑 (여기에 누락 항목이 추가됩니다)
PROXY_MAP: Dict[str, ProxySpec] = {
    # 예시 - 이미 매핑된 항목들(필요 시 수정)
    "IAU": ProxySpec("FRED", "GOLDPMGBD228NLBM", "LBMA Gold Price: PM (USD)"),
    "BCI": ProxySpec("FRED", "SPGSCITR", "S&P GSCI Total Return (proxy for BCOMTR)"),
}

# ------------------------------
# 유틸: ETF 메타데이터 (yfinance)
# ------------------------------
@cache_data
def yf_get_info(ticker: str) -> dict:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        # 일부 필드만 슬라이스
        fields = {
            "longName": info.get("longName"),
            "shortName": info.get("shortName"),
            "category": info.get("category"),
            "quoteType": info.get("quoteType"),
            "fundFamily": info.get("fundFamily"),
            "underlyingSymbol": info.get("underlyingSymbol"),
            "underlyingIndex": info.get("underlyingIndex"),
            "firstTradeDateEpochUtc": info.get("firstTradeDateEpochUtc"),
            "currency": info.get("currency"),
        }
        return {k: v for k, v in fields.items() if v is not None}
    except Exception:
        return {}

# ------------------------------
# 규칙 기반 매퍼: 키워드 → 프록시 제안
# ------------------------------
# FRED 코드를 중심으로 무료 접근이 가능한 지수 우선
# (정밀도 향상을 원하면 유료 소스로 대체 가능)
ASSET_RULES: List[Tuple[re.Pattern, ProxySpec]] = [
    # 금/귀금속
    (re.compile(r"gold|금", re.I), ProxySpec("FRED", "GOLDPMGBD228NLBM", "LBMA Gold Price: PM (USD)")),
    # 은(실버) 예시
    (re.compile(r"silver|은", re.I), ProxySpec("FRED", "SLVPRUSD", "London Fixing Price of Silver (USD/oz)")),
    # 원자재/상품지수(광범위)
    (re.compile(r"commodity|원자재|bloomberg commodity|bcom|gsci", re.I), ProxySpec("FRED", "SPGSCITR", "S&P GSCI Total Return (broad commodity proxy)")),
    # 에너지(유가)
    (re.compile(r"crude|oil|원유|브렌트", re.I), ProxySpec("FRED", "DCOILWTICO", "WTI Crude Oil Price (Spot)", "ffill_only")),
    # 인플레이션 보호채(TIPS)
    (re.compile(r"tips|inflation\-protected|물가연동", re.I), ProxySpec("YF", "TIP", "iShares TIPS ETF (as proxy index)")),
    # 미국 종합채권
    (re.compile(r"agg|total bond|종합채", re.I), ProxySpec("YF", "AGG", "iShares Core U.S. Aggregate Bond (proxy)")),
    # 미국 중기국채(7-10y)
    (re.compile(r"7-10|중기국채|intermediate treasury", re.I), ProxySpec("YF", "IEF", "iShares 7-10y U.S. Treasury (proxy)")),
    # 미국 장기국채
    (re.compile(r"long treasury|장기국채|20\+", re.I), ProxySpec("YF", "TLT", "iShares 20+ Year Treasury (proxy)")),
    # 미국 주식 광범위
    (re.compile(r"s&p|sp500|large cap|미국 주식", re.I), ProxySpec("YF", "SPY", "S&P 500 (via SPY as proxy)")),
    (re.compile(r"total market|전체 시장|미국 전체", re.I), ProxySpec("YF", "VTI", "Vanguard Total U.S. Stock (proxy)")),
    # 리츠(REITs)
    (re.compile(r"reit|리츠", re.I), ProxySpec("YF", "VNQ", "Vanguard U.S. REIT (proxy)")),
    # 금리 단기/현금성
    (re.compile(r"t\-bill|3m|단기국채|현금", re.I), ProxySpec("YF", "BIL", "SPDR 1-3 Month T-Bill (proxy)")),
]

# ------------------------------
# 누락 프록시 감지 & 자동 매핑
# ------------------------------

def detect_missing_proxy(ticker: str, proxy_map: Dict[str, ProxySpec]) -> bool:
    return ticker not in proxy_map


def guess_proxy_for_ticker(ticker: str) -> Optional[ProxySpec]:
    """yfinance 메타데이터 기반 키워드 매칭으로 합리적 프록시 제안."""
    info = yf_get_info(ticker)
    text = " ".join([str(v) for v in info.values()]).lower()
    # 빠른 예외 처리: 금/IAU 같은 케이스는 이름에 gold가 자주 등장
    for pattern, spec in ASSET_RULES:
        if pattern.search(text):
            return spec
    # 실패 시 None
    return None


def audit_and_autofix_proxies(etf_tickers: List[str], proxy_map: Optional[Dict[str, ProxySpec]] = None) -> Tuple[pd.DataFrame, Dict[str, ProxySpec]]:
    """
    전체 티커를 점검하고, 누락 프록시에 대해 규칙 기반 프록시를 자동 제안/적용.
    반환: (리포트 DataFrame, 갱신된 PROXY_MAP)
    """
    pmap = dict(proxy_map or PROXY_MAP)
    records = []

    for tkr in etf_tickers:
        info = yf_get_info(tkr)
        long = info.get("longName") or info.get("shortName") or ""
        underlying = info.get("underlyingIndex") or info.get("underlyingSymbol") or ""
        has_proxy = tkr in pmap

        if has_proxy:
            records.append({
                "티커": tkr,
                "이름": long,
                "기존_매핑": f"{pmap[tkr].source}:{pmap[tkr].series}",
                "상태": "OK",
                "자동제안": "-",
            })
            continue

        # 누락 → 자동 제안 시도
        suggestion = None
        # 1) 발라내기: underlyingIndex 문자열로 직접 감지 가능할 때 (예: NASDAQ 100, S&P 500 etc.)
        if underlying:
            text = underlying.lower()
            # 간단한 케이스 매핑 (필요시 확장)
            if "nasdaq" in text and "100" in text:
                suggestion = ProxySpec("YF", "QQQ", "NASDAQ-100 via QQQ (proxy)")
            elif "s&p" in text and "500" in text:
                suggestion = ProxySpec("YF", "SPY", "S&P 500 via SPY (proxy)")
            elif "treasury" in text and ("7-10" in text or "intermediate" in text):
                suggestion = ProxySpec("YF", "IEF", "U.S. 7-10y Treasury (proxy)")

        # 2) 메타데이터 키워드 규칙
        if suggestion is None:
            suggestion = guess_proxy_for_ticker(tkr)

        if suggestion is not None:
            pmap[tkr] = suggestion
            records.append({
                "티커": tkr,
                "이름": long,
                "기존_매핑": "(없음)",
                "상태": "AUTO_MAPPED",
                "자동제안": f"{suggestion.source}:{suggestion.series} ({suggestion.name})",
            })
        else:
            records.append({
                "티커": tkr,
                "이름": long,
                "기존_매핑": "(없음)",
                "상태": "NEEDS_MANUAL",
                "자동제안": "(규칙 매칭 실패)",
            })

    report = pd.DataFrame(records)
    return report, pmap

# ------------------------------
# JSON 저장/불러오기 (지속성)
# ------------------------------

def save_proxy_map_json(pmap: Dict[str, ProxySpec], path: str) -> None:
    obj = {k: asdict(v) for k, v in pmap.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_proxy_map_json(path: str) -> Dict[str, ProxySpec]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {k: ProxySpec(**v) for k, v in obj.items()}

# ------------------------------
# (선택) Streamlit Admin 패널
# ------------------------------

def render_admin_audit_panel():
    st.markdown("### 🛠️ ETF 추종지수 자동 점검 & 프록시 매핑")
    tickers = st.text_area("ETF 티커 목록 (쉼표/공백 구분)", value="IAU, BCI, QQQ, IEF, TIP").strip()
    if st.button("점검 실행"):
        etf_list = re.split(r"[\s,]+", tickers)
        report, new_map = audit_and_autofix_proxies(etf_list, PROXY_MAP)
        st.success("점검 완료")
        st.dataframe(report, use_container_width=True)
        # 저장 옵션
        col1, col2 = st.columns(2)
        with col1:
            if st.button("PROXY_MAP.json 저장"):
                save_proxy_map_json(new_map, "PROXY_MAP.json")
                st.info("파일로 저장했습니다: PROXY_MAP.json")
        with col2:
            st.download_button("현재 PROXY_MAP 내려받기(JSON)", data=json.dumps({k: asdict(v) for k, v in new_map.items()}, ensure_ascii=False, indent=2), file_name="PROXY_MAP.json", mime="application/json")

# 모듈 단독 실행 시 간단 테스트
if __name__ == "__main__":
    sample = ["IAU", "BCI", "QQQ", "IEF", "TIP", "VNQ", "VTI"]
    rep, newmap = audit_and_autofix_proxies(sample, PROXY_MAP)
    print(rep)
    save_proxy_map_json(newmap, "PROXY_MAP.json")
    print("Saved PROXY_MAP.json")
