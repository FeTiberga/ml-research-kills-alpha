# ml_research_kills_alpha/reports/generate_latex.py
# TODO: Very early version, needs to be improved.

import os
import pandas as pd

def dataframe_to_latex(df: pd.DataFrame, out_path: str, caption: str, label: str):
    with open(out_path, "w") as f:
        f.write(df.to_latex(index=False, caption=caption, label=label))

def main():
    os.makedirs("reports/latex", exist_ok=True)
    df = pd.read_csv("reports/tables/strategy_performance.csv")
    dataframe_to_latex(df,
                       "reports/latex/strategy_performance.tex",
                       caption="Strategy Performance",
                       label="tab:perf")
    # …etc…

if __name__ == "__main__":
    main()
