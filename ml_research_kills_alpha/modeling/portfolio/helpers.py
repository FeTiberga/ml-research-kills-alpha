# ml_research_kills_alpha/portfolio/helpers.py
import numpy as np
import pandas as pd

ID_COLS   = {'permno','date'}
META_COLS = {'ret','tbl','prc','shrout','Size','exchcd','shrcd','eff_half_spread'}
RESERVED  = ID_COLS | META_COLS | {'y_hat','rx_fwd','ret_fwd','rf','mcap','yyyymm'}

def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Basic prep: monthly period, mcap, next-month realized excess.
    
    Args:
        panel (pd.DataFrame): raw CRSP monthly data with columns:
            'permno', 'date', 'ret', 'retx', 'tbl', 'prc', 'shrout'
    """
    df = panel.copy()
    # Ensure monthly period
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
    # Market cap (CRSP: shrout in thousands; prc may be negative)
    df['mcap'] = df['prc'].abs() * df['shrout'] * 1000.0
    # Risk-free as monthly decimal
    df['rf'] = df.groupby('date')['tbl'].transform('first')
    # Align realized one-month-ahead returns
    df = df.sort_values(['permno','date'])
    df['ret_fwd'] = df.groupby('permno')['ret'].shift(-1)
    df['retx_fwd'] = df.groupby('permno')['retx'].shift(-1)
    df['rf_fwd']  = df.groupby('permno')['rf'].shift(-1)
    df['rx_fwd']  = df['ret_fwd'] - df['rf_fwd']
    return df

def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    """Heuristic: use numeric columns not in RESERVED as features."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    X_cols = [c for c in num_cols if c not in RESERVED]
    # (Optional) drop ultra-sparse columns
    good = []
    for c in X_cols:
        if df[c].notna().mean() > 0.98:
            good.append(c)
    return good

def build_lstm_sequences(df: pd.DataFrame, X_cols: list[str], seq_len: int = 12):
    """
    Build (X_seq, y, keys) for LSTM training on month-level data.
    For each permno, roll a window of length seq_len ending at month t to predict rx_fwd at t.
    We'll later use these at formation t to predict t+1.
    """
    seq_X, seq_y, keys = [], [], []
    for pid, g in df.sort_values('date').groupby('permno'):
        g = g.copy()
        # Need rx_fwd as target at the sequence END date
        # Ensure we have no NA in features for the window
        mat = g[X_cols + ['rx_fwd']].to_numpy()
        dates = g['date'].values
        for i in range(seq_len-1, len(g)):
            X_win = mat[i-seq_len+1:i+1, :-1]
            y_t   = mat[i, -1]
            if np.isfinite(X_win).all() and np.isfinite(y_t):
                seq_X.append(X_win)
                seq_y.append(y_t)
                keys.append((pid, dates[i]))   # sequence ends at month t (formation month)
    X_seq = np.array(seq_X, dtype=float)
    y = np.array(seq_y, dtype=float)
    keys = pd.DataFrame(keys, columns=['permno','date'])
    return X_seq, y, keys
