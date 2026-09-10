"""Top-level Options Terminal Quant Engine orchestrator."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
import pandas as pd

from .models import (
    Convention,
    GexProfile,
    OptionContract,
    OptionsTerminalResult,
    RealizedVolCones,
    VolTermStructure,
)
from .greeks import calculate_greeks, vectorized_greeks
from .gex import clean_chain, compute_gex_profile, find_gamma_flip
from .vol import compute_vol_term_structure, compute_25d_skew, build_surface_grid
from .cones import compute_vol_cones_and_vrp, compute_yang_zhang_vol
from .storage import init_options_db, record_snapshot, get_snapshot_history, get_history_status

__all__ = [
    "Convention",
    "OptionContract",
    "GexProfile",
    "VolTermStructure",
    "RealizedVolCones",
    "OptionsTerminalResult",
    "calculate_greeks",
    "vectorized_greeks",
    "clean_chain",
    "compute_gex_profile",
    "find_gamma_flip",
    "compute_vol_term_structure",
    "compute_25d_skew",
    "build_surface_grid",
    "compute_vol_cones_and_vrp",
    "compute_yang_zhang_vol",
    "init_options_db",
    "record_snapshot",
    "get_snapshot_history",
    "get_history_status",
    "compute_options_terminal",
]


def compute_options_terminal(
    ticker: str,
    spot: float,
    chain_df: Optional[pd.DataFrame],
    daily_df: Optional[pd.DataFrame],
    convention: Convention = "naive",
    rf_rate: float = 0.045,
    record_db: bool = False,
) -> OptionsTerminalResult:
    """Compute complete institutional options suite for a ticker.

    Integrates GEX profile with 0DTE volume weighting, dual dealer-sign conventions,
    volatility term structure, 3D surface grid, Yang-Zhang volatility cones, and VRP.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # If options chain is unavailable (e.g. non-optionable stock)
    if chain_df is None or chain_df.empty:
        # Fall back gracefully to Realized Volatility Cones from daily_prices
        cones = compute_vol_cones_and_vrp(daily_df, current_iv=25.0)
        return OptionsTerminalResult(
            ticker=ticker.upper(),
            spot_price=spot,
            as_of=now_utc,
            convention=convention,
            gex=GexProfile(
                total_gex=0.0, call_gex=0.0, put_gex=0.0, regime="positive",
                gamma_flip=None, call_wall=None, call_wall_val=0.0,
                put_wall=None, put_wall_val=0.0, hvl=None, hvl_val=0.0,
                convention=convention,
            ),
            vol=VolTermStructure(),
            cones=cones,
            contracts=[],
            error=f"Options chain is currently unavailable for {ticker.upper()}.",
        )

    try:
        # 1. Clean and normalize chain
        cleaned_df = clean_chain(chain_df)

        # 2. Compute GEX Profile & Structural Levels
        gex = compute_gex_profile(cleaned_df, spot, convention=convention, r=rf_rate)

        # 3. Compute Volatility Term Structure & 3D Surface
        vol = compute_vol_term_structure(cleaned_df, spot)

        # 4. Front ATM IV for VRP calculation
        front_atm_iv = vol.atm_ivs[0] if vol.atm_ivs else (cleaned_df["iv"].median() * 100.0)

        # 5. Compute Realized Volatility Cones, Yang-Zhang, & 68% Coverage
        cones = compute_vol_cones_and_vrp(daily_df, current_iv=front_atm_iv)

        # 6. Format tabular contracts for Greek matrix
        contracts_list = []
        # Group by strike for matrix view
        strikes = sorted(cleaned_df["strike"].unique())
        for k in strikes:
            sub = cleaned_df[cleaned_df["strike"] == k]
            c_row = sub[sub["cp"] == "C"]
            p_row = sub[sub["cp"] == "P"]
            
            row_dict = {
                "strike": float(k),
                "call_bid": float(c_row["bid"].iloc[0]) if not c_row.empty else None,
                "call_ask": float(c_row["ask"].iloc[0]) if not c_row.empty else None,
                "call_iv": round(float(c_row["iv"].iloc[0]) * 100, 1) if not c_row.empty else None,
                "call_oi": int(c_row["open_interest"].iloc[0]) if not c_row.empty else 0,
                "call_vol": int(c_row["volume"].iloc[0]) if not c_row.empty else 0,
                "call_delta": round(float(c_row["delta"].iloc[0]), 3) if not c_row.empty and "delta" in c_row.columns else None,
                "call_gamma": round(float(c_row["gamma"].iloc[0]), 4) if not c_row.empty and "gamma" in c_row.columns else None,
                "call_vanna": round(float(c_row["vanna"].iloc[0]), 4) if not c_row.empty and "vanna" in c_row.columns else None,
                "put_bid": float(p_row["bid"].iloc[0]) if not p_row.empty else None,
                "put_ask": float(p_row["ask"].iloc[0]) if not p_row.empty else None,
                "put_iv": round(float(p_row["iv"].iloc[0]) * 100, 1) if not p_row.empty else None,
                "put_oi": int(p_row["open_interest"].iloc[0]) if not p_row.empty else 0,
                "put_vol": int(p_row["volume"].iloc[0]) if not p_row.empty else 0,
                "put_delta": round(float(p_row["delta"].iloc[0]), 3) if not p_row.empty and "delta" in p_row.columns else None,
                "put_gamma": round(float(p_row["gamma"].iloc[0]), 4) if not p_row.empty and "gamma" in p_row.columns else None,
                "put_charm": round(float(p_row["charm"].iloc[0]), 4) if not p_row.empty and "charm" in p_row.columns else None,
            }
            contracts_list.append(row_dict)

        # 7. Optionally record snapshot into SQLite
        if record_db:
            record_snapshot(
                symbol=ticker,
                spot=spot,
                atm_iv=front_atm_iv,
                hv30=cones.hv30 or cones.yang_zhang_30,
                vrp=cones.vrp,
                call_wall=gex.call_wall,
                put_wall=gex.put_wall,
                gamma_flip=gex.gamma_flip,
                total_gex=gex.total_gex,
                skew_25d=vol.skew_25d,
                convention=convention,
            )

        return OptionsTerminalResult(
            ticker=ticker.upper(),
            spot_price=spot,
            as_of=now_utc,
            convention=convention,
            gex=gex,
            vol=vol,
            cones=cones,
            contracts=contracts_list,
            error=None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        cones = compute_vol_cones_and_vrp(daily_df, current_iv=25.0)
        return OptionsTerminalResult(
            ticker=ticker.upper(),
            spot_price=spot,
            as_of=now_utc,
            convention=convention,
            gex=GexProfile(
                total_gex=0.0, call_gex=0.0, put_gex=0.0, regime="positive",
                gamma_flip=None, call_wall=None, call_wall_val=0.0,
                put_wall=None, put_wall_val=0.0, hvl=None, hvl_val=0.0,
                convention=convention,
            ),
            vol=VolTermStructure(),
            cones=cones,
            contracts=[],
            error=f"Error analyzing options chain: {str(e)}",
        )
