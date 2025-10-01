# ml_research_kills_alpha/portfolio/helpers.py
from typing import Tuple
import numpy as np
import pandas as pd

from ml_research_kills_alpha.support import Logger

ID_COLS   = {'permno','date'}
META_COLS = {'ret','tbl','prc','shrout','Size','exchcd','shrcd','eff_half_spread'}
RESERVED  = ID_COLS | META_COLS | {'y_hat','rx_fwd','ret_fwd','rf','mcap','yyyymm'}
COST_COL = 'BidAskSpread'


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
        if COST_COL not in self.panel.columns:
            self.panel[COST_COL] = 0.0  # you can merge real Chen–Velikov later

        self.panel = self.panel[['permno', 'date', self.target_column, 'me', COST_COL]]
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

    def assign_deciles(self, month_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Assign deciles to the predictions of the specified model.
        """
        self.logger.info("Assigning deciles based on model predictions")
        
        # Ensure the model's prediction column exists
        prediction_col = f'y_hat_{model_name}'
        if prediction_col not in month_df.columns:
            self.logger.error(f"Predictions for model '{model_name}' not found in DataFrame.")
            raise ValueError(f"Predictions for model '{model_name}' not found in DataFrame.")
        
        df = month_df.copy()
        series = df[prediction_col].dropna()

        deciles = np.arange(0.1, 1.0, 0.1)
        cutpoint = series.quantile(deciles).values
        edges = np.concatenate(([-np.inf], cutpoint, [np.inf]))
        df['decile'] = pd.cut(df[prediction_col], bins=edges, labels=np.arange(1, 11), include_lowest=True).astype(int)
        return df

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
    
    def realized_returns_for_month(self, panel: pd.DataFrame, date_period: pd.Period) -> pd.Series:
        """
        Get realized simple returns for a specific month.
        """
        month_slice = panel[panel['date'] == date_period][['permno', self.target_column]].copy()
        by_permno = month_slice.set_index('permno')[self.target_column]
        return pd.to_numeric(by_permno, errors='coerce').fillna(0.0)

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

    def backtest_decile_long_short(self, preds_merged: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Monthly decile long-short backtest on the panel.

        Mechanics per formation month t (preds date == t-1 in your trainer, used for month t):
        1) Assign deciles on f'y_hat_{model_name}' using current month cross-section.
        2) Build target long/short legs (D10 / D1) by value weights (sum to +1 each leg).
        3) Drift last month's holdings through realized returns of month t (pre-rebalance).
        4) Two-sided turnover & trading cost at month t using one-way effective half-spread column.
        5) Earn gross in month t+1 using target weights;
                net = (gross_long - cost_long) - (gross_short - cost_short).

        Args:
            preds_merged: Output of merge_preds(). Must contain:
                ['permno','date', self.target_column, 'me', COST_COL, f'y_hat_{model_name}'].
            model_name: Name used to resolve f'y_hat_{model_name}'.

        Returns:
            DataFrame indexed by month-end date with:
                ['gross_long','gross_short','gross_ls',
                'turnover_long','turnover_short','cost_long','cost_short',
                'net_ls','cum_gross_ls','cum_net_ls'].
        """
        results = []
        previous_long_weights: pd.Series | None = None
        previous_short_weights: pd.Series | None = None

        # We assume preds_merged is already NYSE-only & cleaned.
        preds_merged = preds_merged.sort_values(['date','permno'])

        for formation_month, month_df in preds_merged.groupby('date', sort=True):
            # 1) Sort into deciles
            month_df = self.assign_deciles(month_df, model_name=model_name)

            # 2) Target leg weights at formation t
            long_target_weights, short_target_weights = self.target_weights_month(month_df)

            # 3) Drift prior holdings through returns of month t (pre-rebalance)
            returns_t = self.realized_returns_for_month(self.panel, formation_month, self.target_column)
            long_pre = self.drift(previous_long_weights, returns_t)
            short_pre = self.drift(previous_short_weights, returns_t)

            # 4) Turnover & trading costs using effective half-spread at month t
            half_spread_t = month_df.set_index('permno')[COST_COL]
            long_to, long_cost = self.turnover_and_cost(long_target_weights, long_pre, half_spread_t)
            short_to, short_cost = self.turnover_and_cost(short_target_weights, short_pre, half_spread_t)

            # 5) Gross return in month t+1 (payoff month)
            payoff_month = formation_month + 1
            returns_t1 = self.realized_returns_for_month(self.panel, payoff_month, self.target_column)

            gross_long = (long_target_weights * returns_t1.reindex(long_target_weights.index).fillna(0.0)).sum()
            gross_short = (short_target_weights * returns_t1.reindex(short_target_weights.index).fillna(0.0)).sum()
            gross_ls = gross_long - gross_short
            net_ls = (gross_long - long_cost) - (gross_short - short_cost)

            results.append({
                'date': formation_month.to_timestamp('M'),
                'gross_long': gross_long,
                'gross_short': gross_short,
                'gross_ls': gross_ls,
                'turnover_long': long_to,
                'turnover_short': short_to,
                'cost_long': long_cost,
                'cost_short': short_cost,
                'net_ls': net_ls,
            })

            previous_long_weights = long_target_weights
            previous_short_weights = short_target_weights

        res = pd.DataFrame(results).set_index('date').sort_index()
        res['cum_gross_ls'] = (1.0 + res['gross_ls']).cumprod()
        res['cum_net_ls']   = (1.0 + res['net_ls']).cumprod()
        return res
    
    @staticmethod
    def summarize_backtest(monthly_results: pd.DataFrame) -> dict:
        """
        Summarize monthly long-short results.

        Args:
            monthly_results: Output from backtest_decile_long_short().

        Returns:
            Dict with mean %, t-stats, annualized Sharpe, avg turnover/costs, and N months.
        """
        def ann_sharpe(x: pd.Series) -> float:
            mu, sd = x.mean(), x.std(ddof=1)
            return float((mu/sd)*np.sqrt(12.0)) if sd > 0 else np.nan

        def tstat(x: pd.Series) -> float:
            mu, sd, n = x.mean(), x.std(ddof=1), len(x)
            return float(mu/(sd/np.sqrt(n))) if sd > 0 else np.nan

        res = monthly_results
        return {
            'avg_gross_%': 100.0 * res['gross_ls'].mean(),
            'tstat_gross': tstat(res['gross_ls']),
            'sharpe_gross': ann_sharpe(res['gross_ls']),
            'avg_net_%': 100.0 * res['net_ls'].mean(),
            'tstat_net': tstat(res['net_ls']),
            'sharpe_net': ann_sharpe(res['net_ls']),
            'avg_turnover_long_%': 100.0 * res['turnover_long'].mean(),
            'avg_turnover_short_%': 100.0 * res['turnover_short'].mean(),
            'avg_cost_long_bp': 1e4 * res['cost_long'].mean(),
            'avg_cost_short_bp': 1e4 * res['cost_short'].mean(),
            'N_months': len(res),
        }
