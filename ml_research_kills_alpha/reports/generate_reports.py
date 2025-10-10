# Create a reusable Python script that reads the uploaded data, makes figures similar to the examples,
# and generates a LaTeX table from the summary CSV.
#
# Notes (tool rules):
# - Using matplotlib (no seaborn).
# - One chart per figure.
# - Do not set specific colors.

from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import dedent

from ml_research_kills_alpha.config import REPORTS_DIR, MODELS_DIR

INPUT_DIR = MODELS_DIR / "backtests"
OUTPUT_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_backtests(model_files: dict) -> dict:
    """
    Load backtest CSV files into a dictionary of DataFrames keyed by model name.

    Args:
        model_files (dict): Mapping from model name (str) to CSV filepath (str or Path).

    Returns:
        dict: Mapping from model name (str) to pandas.DataFrame with a parsed 'date' column.
    """
    data = {}
    for model, fp in model_files.items():
        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        data[model] = df
    return data


def compute_sharpe_ratio(monthly_returns: pd.Series) -> float:
    """
    Compute annualized Sharpe ratio from a series of monthly returns (excess returns).

    Args:
        monthly_returns (pd.Series): Monthly excess returns as decimals (e.g., 0.012 for 1.2%).

    Returns:
        float: Annualized Sharpe ratio.
    """
    mu = monthly_returns.mean()
    sigma = monthly_returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    monthly_sr = mu / sigma
    annualized_sr = monthly_sr * np.sqrt(12.0)
    return float(annualized_sr)


def plot_cumulative_net_ls(m, df) -> Path:
    """
    Plot cumulative net long-short performance for all provided models in a publication-style line chart.

    The function mirrors the style of the user's example:
    - Clean grid with dashed y-grid
    - Different line styles/markers per series (no explicit color settings)
    - Legend box with model names
    - Dates shown along the x-axis

    Args:
        backtests (dict): Mapping model name -> DataFrame containing 'date' and either 'cum_net_ls'
                          or 'net_ls' (monthly net long-short). If only 'net_ls' is present, a
                          cumulative series is constructed assuming $1 initial capital.
        outfile (Path): Output path for the saved figure (PDF).

    Returns:
        Path: Path to the saved figure.
    """
    plt.figure(figsize=(7.0, 4.5))
    outfile = OUTPUT_DIR / f"cumulative_ls_{m}.pdf"

    # Assign repeating styles to differentiate lines without specifying colors
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    markers = ["", "o", "s", "D", "^", "v", "x", "+", "*", "P"]

    y_net = df["cum_net_ls"]
    y_gross = df["cum_gross_ls"]

    style = line_styles[0 % len(line_styles)]
    marker = markers[0 % len(markers)]
    plt.plot(df["date"], y_gross, linestyle=style, marker=marker, markevery=max(1, len(df)//24), linewidth=1.6, label=m + " (Gross)")

    style = line_styles[1 % len(line_styles)]
    marker = markers[1 % len(markers)]
    plt.plot(df["date"], y_net, linestyle=style, marker=marker, markevery=max(1, len(df)//24), linewidth=1.6, label=m + " (Net)")

    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.title("Gross / Net cumulative performance of long-short strategies", pad=10)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(frameon=True, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile


def plot_two_best_with_sharpe(backtests: dict, summary_df: pd.DataFrame, outfile: Path) -> Path:
    """
    Plot a comparison chart of the two best strategies by net Sharpe ratio and annotate
    each line with its Sharpe ratio, mimicking the user's example annotation style.

    Args:
        backtests (dict): Mapping model name -> DataFrame with 'date' and 'cum_net_ls' or 'net_ls'.
        summary_df (pd.DataFrame): Summary table with 'model' and 'sharpe_net' columns.
        outfile (Path): Output path for the saved figure (PDF).

    Returns:
        Path: Path to the saved figure.
    """
    # Choose top two by sharpe_net (descending)
    top2 = summary_df.sort_values("sharpe_net", ascending=False).head(2)
    selected_models = top2["model"].tolist()

    plt.figure(figsize=(7.0, 4.5))

    styles = ["-", "--"]
    markers = ["o", "s"]
    for i, model in enumerate(selected_models):
        df = backtests[model]
        if "cum_net_ls" in df.columns:
            y = df["cum_net_ls"]
        else:
            series = df["net_ls"]
            monthly = series/100.0 if series.abs().median() > 1.0 else series
            y = (1.0 + monthly).cumprod()
        plt.plot(df["date"], y, linestyle=styles[i % len(styles)], marker=markers[i % len(markers)],
                 markevery=max(1, len(df)//20), linewidth=1.8, label=model)

    # Annotate Sharpe ratios
    for _, row in top2.iterrows():
        model = row["model"]
        sharpe = row["sharpe_net"]
        text = f"Sharpe ratio ({model}) = {sharpe:.3f}"
        # place text in axes fraction coordinates to avoid overlapping the lines
        # positions spaced vertically
        y_frac = 0.90 - 0.08 * (_ % 2)
        plt.gca().text(0.02, y_frac, text, transform=plt.gca().transAxes, fontsize=9)

    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.title("Comparison of two best net strategies", pad=10)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(frameon=True, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile


def add_sig_stars(tstat: float) -> str:
    """
    Convert a t-statistic to significance stars.
    Uses |t| thresholds ~ 1.96 (5%), 2.58 (1%), 3.29 (0.1%).

    Args:
        tstat (float): t-statistic.

    Returns:
        str: A string of asterisks representing significance ('', '*', '**', '***').
    """
    at = abs(tstat)
    if at >= 3.29:
        return r"\sym{***}"
    if at >= 2.58:
        return r"\sym{**}"
    if at >= 1.96:
        return r"\sym{*}"
    return ""


def latex_summary_table(summary_df: pd.DataFrame, outfile: Path) -> Path:
    """
    Build a LaTeX table (booktabs style) from the summary DataFrame.

    Columns included:
        Model, Gross avg% [t], SR; Net avg% [t], SR; Turnover long%; Turnover short%; Cost long (bp); Cost short (bp)

    Args:
        summary_df (pd.DataFrame): DataFrame with expected columns (see summary.csv).
        outfile (Path): File path where the LaTeX code will be written.

    Returns:
        Path: Path to the saved .tex file.
    """
    df = summary_df.copy()

    # Order models by net Sharpe ratio descending
    df = df.sort_values("sharpe_net", ascending=False).reset_index(drop=True)

    # Format rows
    rows = []
    for _, r in df.iterrows():
        gross = f"{r['avg_gross_%']:.2f}"
        t_g = f"[{r['tstat_gross']:.2f}]"
        sr_g = f"{r['sharpe_gross']:.2f}"

        net = f"{r['avg_net_%']:.2f}"
        t_n = f"[{r['tstat_net']:.2f}]"
        sr_n = f"{r['sharpe_net']:.2f}"

        # attach stars to avg columns (based on t-stats)
        gross_star = add_sig_stars(r["tstat_gross"])
        net_star = add_sig_stars(r["tstat_net"])

        row = [
            r["model"],
            f"{gross}{gross_star}",
            t_g,
            sr_g,
            f"{net}{net_star}",
            t_n,
            sr_n,
            f"{r['avg_turnover_long_%']:.2f}",
            f"{r['avg_turnover_short_%']:.2f}",
            f"{r['avg_cost_long_bp']:.2f}",
            f"{r['avg_cost_short_bp']:.2f}",
        ]
        rows.append(" & ".join(row) + r" \\")

    body = "\n".join(rows)

    latex = dedent(
        r"""
        % Table generated by ChatGPT: publication-style summary of model performance
        \begin{table}[!ht]
        \centering
        \caption{Out-of-sample performance summary (gross vs. net)}
        \label{tab:summary_performance}
        \begingroup
        \def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}
        \setlength{\tabcolsep}{6pt}
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{lrrrrrrrrrr}
        \toprule
        & \multicolumn{3}{c}{Gross} & \multicolumn{3}{c}{Net} & \multicolumn{2}{c}{Turnover (\%)} & \multicolumn{2}{c}{Cost (bp)}\\
        \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
        Model & Avg.\% & {[t]} & SR & Avg.\% & {[t]} & SR & Long & Short & Long & Short\\
        \midrule
        """
    ).strip("\n")

    latex_end = dedent(
        r"""
        \bottomrule
        \end{tabular}
        \par\smallskip \footnotesize Notes: Avg.% are average monthly excess returns in percent. SR = annualized Sharpe ratio.
        Stars indicate significance by absolute t-stat: \sym{*} $>$ 1.96, \sym{**} $>$ 2.58, \sym{***} $>$ 3.29.
        \endgroup
        \end{table}
        """
    ).strip("\n")

    table_tex = latex + "\n" + body + "\n" + latex_end

    outfile.write_text(table_tex)
    return outfile


# ---- Run everything on the uploaded data ----

"""model_files = {
    "ENET": INPUT_DIR / "backtest_ENET.csv",
    "Ensemble": INPUT_DIR / "backtest_Ensemble.csv",
    "FFNN2": INPUT_DIR / "backtest_FFNN2.csv",
    "FFNN3": INPUT_DIR / "backtest_FFNN3.csv",
    "FFNN4": INPUT_DIR / "backtest_FFNN4.csv",
    "FFNN5": INPUT_DIR / "backtest_FFNN5.csv",
    "OLS-H": INPUT_DIR / "backtest_OLS-H.csv",
}"""

model_files = {
    "ENET": INPUT_DIR / "backtest_ENET.csv",
    "OLS-H": INPUT_DIR / "backtest_OLS-H.csv",
}

backtests = read_backtests(model_files)
summary_df = pd.read_csv(INPUT_DIR / "summary.csv")

# Figure 1: all models cumulative net LS
for m, df in backtests.items():
    plot_cumulative_net_ls(m, df)

# Figure 2: two best by net Sharpe
fig2 = OUTPUT_DIR / "comparison_two_best_net_sharpe.pdf"
plot_two_best_with_sharpe(backtests, summary_df, fig2)

# Table: LaTeX summary from summary.csv
tex_out = OUTPUT_DIR / "summary_performance_table.tex"
latex_summary_table(summary_df, tex_out)
