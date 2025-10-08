import numpy as np
import pandas as pd

# ---- Add to Portfolio --------------------------------------------------------

def _norm_to_percentile(x: pd.Series) -> pd.Series:
    """
    Convert [-1, 1]-normalized (percentile/rank) feature to percentile in (0,1).
    Assumes x = 2*(rank-0.5). Inverse: rank = 0.5*(x+1).
    """
    p = 0.5 * (pd.to_numeric(x, errors="coerce") + 1.0)
    return p.clip(1e-4, 1 - 1e-4)

def _percentile_to_halfspread_decimal(p: pd.Series,
                                      anchors_p=(0.05, 0.50, 0.95),
                                      anchors_bp=(5.0, 12.0, 50.0)) -> pd.Series:
    """
    Map percentiles to one-way effective half-spread **decimals** via log-linear interpolation.

    Args:
        p: Percentiles in (0,1).
        anchors_p: Tuple of three anchor percentiles (low, mid, high).
        anchors_bp: Tuple of three anchor half-spread values in **basis points** at those percentiles.

    Returns:
        Series of half-spreads in **decimals** (e.g., 0.0015 = 15 bps).
    """
    p = p.clip(1e-4, 1 - 1e-4)
    p1, p2, p3 = anchors_p
    b1, b2, b3 = anchors_bp

    # work in log-bps for smooth monotone interpolation
    lb1, lb2, lb3 = np.log([b1, b2, b3])

    out = pd.Series(index=p.index, dtype=float)

    # piece 1: [p_min, p1] -> extrapolate to p1
    m12 = (lb2 - lb1) / (p2 - p1)
    m23 = (lb3 - lb2) / (p3 - p2)

    mask1 = (p <= p1)
    mask2 = (p > p1) & (p <= p2)
    mask3 = (p > p2) & (p <= p3)
    mask4 = (p > p3)

    out.loc[mask1] = lb1 + m12 * (p.loc[mask1] - p1)            # extrapolate downwards
    out.loc[mask2] = lb1 + m12 * (p.loc[mask2] - p1)            # interpolate low→mid
    out.loc[mask3] = lb2 + m23 * (p.loc[mask3] - p2)            # interpolate mid→high
    out.loc[mask4] = lb3 + m23 * (p.loc[mask4] - p3)            # extrapolate upwards

    half_spread_bp = np.exp(out)                 # back to bps
    half_spread_dec = (half_spread_bp / 1e4)     # -> decimals
    return half_spread_dec.clip(lower=0.0)

def _half_spread_from_feature(month_df: pd.DataFrame,
                              feature_col: str = "BidAskSpread",
                              anchors_p=(0.05, 0.50, 0.95),
                              anchors_bp=(5.0, 12.0, 50.0)) -> pd.Series:
    """
    Build a decimal half-spread **from a normalized feature** by percentile mapping.

    Args:
        month_df: Monthly cross-section with normalized 'BidAskSpread' ∈ [-1, 1].
        feature_col: Name of the normalized feature column.
        anchors_p: Anchor percentiles.
        anchors_bp: Corresponding half-spread bps at anchors.

    Returns:
        pd.Series of decimal half-spread indexed by permno.
    """
    z = month_df[feature_col]
    p = _norm_to_percentile(z)
    hs = _percentile_to_halfspread_decimal(p, anchors_p=anchors_p, anchors_bp=anchors_bp)
    return hs
