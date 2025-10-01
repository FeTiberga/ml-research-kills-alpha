# ml_research_kills_alpha/portfolio/helpers.py
from typing import Tuple
import numpy as np
import pandas as pd

from ml_research_kills_alpha.support import Logger

ID_COLS   = {'permno','date'}
META_COLS = {'ret','tbl','prc','shrout','Size','exchcd','shrcd','eff_half_spread'}
RESERVED  = ID_COLS | META_COLS | {'y_hat','rx_fwd','ret_fwd','rf','mcap','yyyymm'}


class Portfolio():

    def __init__(self, panel: pd.DataFrame, target_column: str):
        self.panel = panel.copy()
        self.logger = Logger()
        self.target_column = target_column
        self.prepare_panel()


    def prepare_panel(self) -> pd.DataFrame:
        """
        Prepare the panel DataFrame for analysis by ensuring correct dtypes and adding necessary columns.
        Adds
        - 'date' as Period (monthly)
        - 'me' (market equity) if not present, computed from 'prc'
        - 'BidAskSpread' if not present, set to 0.0
        """
        
        self.logger.info(" ")
        self.logger.info("Preparing panel data: ")
        self.logger.info(f"Original shape: {self.panel.shape}")
        
        # Ensure 'date' is Period (monthly)
        self.panel['date'] = pd.to_datetime(self.panel['date']).dt.to_period('M')

        # Market cap for value-weighting
        if 'me' not in self.panel.columns:
            self.logger.info(" Computing Market Equity from 'prc' and 'shrout'")
            if {'prc','shrout'}.issubset(self.panel.columns):
                self.panel['me'] = self.panel['prc'].abs() * self.panel['shrout'] * 1000.0  # shrout is thousands in CRSP
            elif 'Size' in self.panel.columns:
                self.panel['me'] = self.panel['Size']
            else:
                raise ValueError("Need 'prc' & 'shrout' or 'Size' (or precomputed 'me') for weights.")

        # Costs column default
        if 'BidAskSpread' not in self.panel.columns:
            self.panel['BidAskSpread'] = 0.0  # you can merge real Chen–Velikov later

        self.panel = self.panel[['permno', 'date', self.target_column, 'me', 'BidAskSpread']]
        self.logger.info(f" Final shape: {self.panel.shape}")
        return self.panel

    def merge_preds(self, preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Merge model predictions into the prepared panel DataFrame.
        """
        self.logger.info(f"Merging predictions from model '{model_name}' into panel")
        p = self.panel.copy()
        q = preds[preds['model'] == model_name].copy()
        q['date'] = q['date'].astype('period[M]')
        df = p.merge(q[['permno','date','y_hat']], on=['permno','date'], how='inner')
        df.rename(columns={'y_hat': f'y_hat_{model_name}'}, inplace=True)
        return df.sort_values(['date','permno'])

    def assign_deciles(self, panel: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Assign deciles to the predictions of the specified model.
        """
        self.logger.info("Assigning deciles based on model predictions")
        qs = panel.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
        edges = np.concatenate(([-np.inf], qs, [np.inf]))
        panel = panel.copy()
        panel['decile'] = pd.cut(panel[f'y_hat_{model_name}'], bins=edges, labels=np.arange(1, 11), include_lowest=True).astype(int)
        return panel
    
    @staticmethod
    def value_weights(me: pd.Series) -> pd.Series:
        """
        Compute value weights from market equity series.

        Args:
            me (pd.Series): Market equity values.

        Returns:
            pd.Series: Value weights.
                - proportional to market equity
                - sum to 1
                - all zeros if sum of market equity not positive
        """
        me = pd.to_numeric(me, errors="coerce").fillna(0.0).clip(lower=0.0)
        s = me.sum()
        return (me / s) if s > 0 else me*0

    def target_weights_month(self, month_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Build value-weighted long and short leg target weights for a single month.

        Args:
            month_df: Cross-section for one month.

        Returns:
            A tuple
            - long_leg_weights: pd.Series of value weights for long leg (decile 10)
            - short_leg_weights: pd.Series of value weights for short leg (decile 10)
        """
        id_col = "permno"
        weight_col = "me"
        
        long_stocks = month_df.loc[month_df["decile"] == 10, [id_col, weight_col]]
        short_stocks = month_df.loc[month_df["decile"] == 1,  [id_col, weight_col]]

        long_leg_weights = self.value_weights(long_stocks.set_index(id_col)[weight_col])
        short_leg_weights = self.value_weights(short_stocks.set_index(id_col)[weight_col])

        # Ensure index names are consistent for downstream merges
        long_leg_weights.index.name = id_col
        short_leg_weights.index.name = id_col
        return long_leg_weights, short_leg_weights

    @staticmethod
    def drift(previous_weights: pd.Series,
              realized_total_returns: pd.Series) -> pd.Series:
        """
        Roll last month's post-trade weights through this month's realized returns
        to obtain pre-rebalance weights for the new month.

        Args:
            previous_weights: Series of last month's post-trade weights indexed by security ID.
            realized_total_returns: Series of simple total returns (e.g., 'ret')
                for the month just ended, indexed by the same security IDs.

        Returns:
            A Series of pre-rebalance weights (same index as the union of inputs) that:
            - are proportional to previous_weights * (1 + realized_total_returns),
            - are renormalized to sum to 1,
            - are all zeros if exposure collapses to 0 or previous_weights is empty.
        """
        if previous_weights is None or previous_weights.empty:
            return pd.Series(dtype=float)

        unified_index = previous_weights.index.union(realized_total_returns.index)
        prior_weights_aligned = previous_weights.reindex(unified_index).fillna(0.0)
        realized_returns_aligned = pd.to_numeric(
            realized_total_returns.reindex(unified_index), errors="coerce"
        ).fillna(0.0)

        post_return_values = prior_weights_aligned * (1.0 + realized_returns_aligned)
        gross_exposure = float(post_return_values.sum())
        if gross_exposure == 0.0:
            return post_return_values * 0.0

        pre_rebalance_weights = post_return_values / gross_exposure
        pre_rebalance_weights.name = "pre_rebalance_weight"
        return pre_rebalance_weights

    @staticmethod
    def turnover_and_cost(target_weights: pd.Series,
                          pre_rebalance_weights: pd.Series,
                          effective_half_spread: pd.Series) -> Tuple[float, float]:
        """
        Compute two-sided turnover and one-way trading cost for a monthly rebalance.

        Turnover and costs follow the standard anomaly backtesting convention:
        - Turnover_t = Σ_i |w_target_i,t - w_pre_i,t|
        - TradingCost_t = Σ_i |w_target_i,t - w_pre_i,t| x half_spread_i,t
          where half_spread is the composite effective bid-ask **half**-spread (decimal).

        Args:
            target_weights: Desired post-rebalance weights at the start of the new month,
                indexed by security ID (sum to 1 for each leg).
            pre_rebalance_weights: Pre-rebalance weights after drifting prior holdings through returns,
                indexed by security ID (sum to 1 for each leg).
            effective_half_spread: Per-security effective half-spread (e.g., Chen-Velikov composite),
                as decimal costs for the trading month, indexed by security ID.

        Returns:
            A tuple (two_sided_turnover, trading_cost):
            - two_sided_turnover: float, Σ |Δw| over the cross-section,
            - trading_cost: float, Σ |Δw| x half_spread_i.
        """
        unified_index = target_weights.index.union(pre_rebalance_weights.index)

        target_aligned = target_weights.reindex(unified_index).fillna(0.0)
        pre_aligned = pre_rebalance_weights.reindex(unified_index).fillna(0.0)

        trade_sizes = (target_aligned - pre_aligned).abs()
        two_sided_turnover = float(trade_sizes.sum())

        half_spread_aligned = pd.to_numeric(
            effective_half_spread.reindex(unified_index), errors="coerce"
        ).fillna(0.0)
        trading_cost = float((trade_sizes * half_spread_aligned).sum())

        return two_sided_turnover, trading_cost
