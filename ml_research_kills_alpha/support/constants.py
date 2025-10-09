START_DATE = "03/01/1957"
END_DATE_2023 = "12/31/2023"
END_DATE_2024 = "12/31/2024"
CUTOFF_2005 = "01/01/2005"

NON_FEATURES = ["permno", "date", "prc", "shrout", "cfacpr", "cfacshr", "ticker", "cusip", "exchcd", "shrcd", "siccd", "yyyymm", "ff49_short"]
META_COLS = ["permno"]
PREDICTED_COL = ["ret", "retx"]

RANDOM_SEED = 1

LSTM_SEQUENCE_LENGTH = 12  # months

MODELS = [
    "OLS-H", "RIDGE", "LASSO", "ENET",   # linear models
    "RF", "XGB",                         # tree-based and boosting
    "FFNN2", "FFNN3", "FFNN4", "FFNN5",  # neural networks
    "LSTM1", "LSTM2"                     # LSTM for sequence data
]