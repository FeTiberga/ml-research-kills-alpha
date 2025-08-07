# ml-research-kills-alpha

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

We measure how much of the ML edge in predicting future stocks survives when accounting for availability of algorithm architectures

## Project Organization

```
├── LICENSE                   <- Open-source license if one is chosen
├── Makefile                  <- Makefile with convenience commands like `make data` or `make train`
├── README.md                 <- The top-level README for developers using this project.
├── data       
│   ├── external              <- Data from third party sources.
│   ├── interim               <- Intermediate data that has been transformed.
│   ├── processed             <- The final, canonical data sets for modeling.
│   └── raw                   <- The original, immutable data dump.
│       
├── docs                      <- A default mkdocs project; see www.mkdocs.org for details
│       
├── models                    <- Trained and serialized models, model predictions, or model summaries
│       
├── notebooks                 <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                the creator's initials, and a short `-` delimited description, e.g.
│                                `1.0-jqp-initial-data-exploration`.
│       
├── pyproject.toml            <- Project configuration file with package metadata for 
│                                src and configuration for tools like black
│       
├── references                <- Data dictionaries, manuals, and all other explanatory materials.
│       
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures               <- Generated graphics and figures to be used in reporting
│       
├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
│                                generated with `pip freeze > requirements.txt`
│       
├── setup.cfg                 <- Configuration file for flake8
│       
└── ml_research_kills_alpha   <- Source code for use in this project.
    │
    ├── __init__.py                <- Makes src a Python module
    │   
    ├── config.py                  <- Store useful variables and configuration
    │   
    ├── datasets                   <- Scripts to download or generate data
    │   ├── __init__.py    
    │   ├── download.py            <- basic class to handle downloading logic   
    │   └── chen_zimmermann.py     <- code to download data from Chen Zimmermann (2020)
    │   └── crsp.py                <- code to download CRSP stock data
    │    
    ├── features.py                <- Code to create features for modeling
    │   
    ├── modeling                   
    │   ├── __init__.py    
    │   ├── predict.py             <- Code to run model inference with trained models          
    │   └── train.py               <- Code to train models
    │   
    └── plots.py                   <- Code to create visualizations
```

--------

