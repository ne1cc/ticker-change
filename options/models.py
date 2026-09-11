"""Data models and type definitions for the options terminal suite."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Convention = Literal["naive", "short_wings"]


@dataclass
class OptionContract:
    strike: float
    dte: int
    expiration: str
    cp: Literal["C", "P"]
    bid: float
    ask: float
    last_price: float
    iv: float
    volume: int
    open_interest: int
    delta: float
    gamma: float
    theta: float
    vega: float
    vanna: float
    charm: float
    gex: float  # In millions $ per 1% spot move


@dataclass
class GexProfile:
    total_gex: float  # Sum of net GEX in $M
    call_gex: float
    put_gex: float
    regime: Literal["positive", "negative"]
    gamma_flip: Optional[float]
    call_wall: Optional[float]
    call_wall_val: float
    put_wall: Optional[float]
    put_wall_val: float
    hvl: Optional[float]
    hvl_val: float
    convention: Convention
    strikes: List[float] = field(default_factory=list)
    net_gex_series: List[float] = field(default_factory=list)
    call_gex_series: List[float] = field(default_factory=list)
    put_gex_series: List[float] = field(default_factory=list)
    vanna_series: List[float] = field(default_factory=list)
    charm_series: List[float] = field(default_factory=list)
    top_nodes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VolTermStructure:
    expirations: List[str] = field(default_factory=list)
    dtes: List[int] = field(default_factory=list)
    atm_ivs: List[float] = field(default_factory=list)
    slope: float = 0.0
    regime: Literal["contango", "backwardation", "flat"] = "contango"
    skew_25d: float = 1.0
    risk_reversal: float = 0.0
    surface_grid: Optional[Dict[str, Any]] = None  # {moneyness: [], dtes: [], iv_grid: [[]]}


@dataclass
class RealizedVolCones:
    current_iv: float
    hv10: float
    hv20: float
    hv30: float
    hv60: float
    hv90: float
    yang_zhang_30: float
    yang_zhang_90: float
    vrp: float  # IV - HV30
    vrp_regime: str  # e.g. "Negative VRP (Underpriced Volatility)"
    iv_rank: float
    iv_percentile: float
    iv_min_52w: float
    iv_max_52w: float
    expected_move_1d: float
    expected_move_straddle: float
    coverage_68_pct: float
    is_accumulating: bool = False
    history_days: int = 0


@dataclass
class OptionsTerminalResult:
    ticker: str
    spot_price: float
    as_of: str
    convention: Convention
    gex: GexProfile
    vol: VolTermStructure
    cones: RealizedVolCones
    contracts: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
