# auto_index_mapper.py
from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class IndexMeta:
    name: str
    provider: str
    proxy_ticker: Optional[str] = None   # 인덱스 직접시계열 없을 때 대체(예: ^NDX 등)
    notes: Optional[str] = None

# 기본 매핑(필요시 계속 보강)
_DEFAULT_MAP: Dict[str, IndexMeta] = {
    # Core US
    "SPY":  IndexMeta("S&P 500", "S&P DJI", "^GSPC", "미상장 구간은 S&P500 지수사용"),
    "IVV":  IndexMeta("S&P 500", "S&P DJI", "^GSPC"),
    "VOO":  IndexMeta("S&P 500", "S&P DJI", "^GSPC"),
    "QQQ":  IndexMeta("NASDAQ-100", "Nasdaq", "^NDX"),
    "VTI":  IndexMeta("CRSP US Total Market", "CRSP", None),
    "IWM":  IndexMeta("Russell 2000", "FTSE Russell", "^RUT"),

    # KB_Portfolio 관련
    "IEF":  IndexMeta("ICE U.S. Treasury 7–10 Year", "ICE BofA/ICE"),
    "TIP":  IndexMeta("Bloomberg U.S. TIPS", "Bloomberg"),
    "VCLT": IndexMeta("Bloomberg U.S. Long Corporate Bond", "Bloomberg"),
    "EMLC": IndexMeta("J.P. Morgan GBI-EM Global Diversified (LC)", "JPMorgan"),
    "GDX":  IndexMeta("NYSE Arca Gold Miners Index", "NYSE Arca"),
    "MOO":  IndexMeta("MVIS Global Agribusiness", "MV Index Solutions"),
    "XLB":  IndexMeta("Materials Select Sector Index", "S&P DJI"),
    "VDE":  IndexMeta("MSCI US IMI Energy 25/50", "MSCI"),
    # 금/원자재 레퍼런스
    "IAU":  IndexMeta("LBMA Gold Price", "LBMA", None),
    "GLD":  IndexMeta("LBMA Gold Price", "LBMA", None),
    "BCI":  IndexMeta("Bloomberg Commodity Index", "Bloomberg"),

    # 기타 예시
    "AGG":  IndexMeta("Bloomberg U.S. Aggregate Bond", "Bloomberg"),
    "LQD":  IndexMeta("Bloomberg U.S. Corporate Bond", "Bloomberg"),
}

CSV_PATHS = [
    "./data/index_mapping.csv",   # 사용자 확장용 (권장)
    "./index_mapping.csv",
]

def _load_csv_maps() -> Dict[str, IndexMeta]:
    result: Dict[str, IndexMeta] = {}
    for p in CSV_PATHS:
        if os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get("ticker","").strip().upper()
                    if not t: 
                        continue
                    result[t] = IndexMeta(
                        name=row.get("index_name","").strip(),
                        provider=row.get("provider","").strip() or "unknown",
                        proxy_ticker=(row.get("proxy_ticker") or "").strip() or None,
                        notes=(row.get("notes") or "").strip() or None,
                    )
    return result

# 외부에서 쓰는 진입점
def build_index_map() -> Dict[str, IndexMeta]:
    user_map = _load_csv_maps()
    merged = dict(_DEFAULT_MAP)
    merged.update(user_map)  # CSV가 우선
    return merged

def auto_map_index(ticker: str) -> Optional[IndexMeta]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    return build_index_map().get(t)
