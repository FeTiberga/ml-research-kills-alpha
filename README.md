# ml-research-kills-alpha

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

We measure how much of the ML edge in predicting future stocks survives when accounting for availability of algorithm architectures

## Project Organization

```
├── LICENSE                   <- Open-source MIT license
├── Makefile                  <- Makefile with convenience commands like `make data` or `make train`
├── README.md                 <- The top-level README for developers using this project.
|
├── data       
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
├── requirements.txt          <- The requirements file for reproducing the analysis environment
│       
├── setup.cfg                 <- Configuration file for flake8
|
└── ml_research_kills_alpha   <- Source code for use in this project.
    │
    ├── __init__.py                <- Makes src a Python module
    │
    ├── config.py                  <- Store useful variables and configuration
    │
    ├── datasets                   <- Scripts to download or generate data
    │   ├── __init__.py
    │   ├── data                                  <- Small data files that can be pushed to Github
    │   │   ├── cz_signal_doc.py                  <- Description and publication date of Chen Zimmermann (2020) signals
    │   │   ├── ff_industry_classification.json   <- 49 industry classification indicators from Fama and French (1997)
    │   │   └── gw_predictor_data_2024            <- Data from Welch and Goyal (2008) up to December 2024
    │   ├── processed                             <- Scripts to process and clean raw datasets
    │   │   ├── clearer.py                        <- Basic class to handle cleaning logic
    │   │   ├── chen_zimmermann.py                <- Code to clean data from Chen Zimmermann (2020)
    │   │   └── crsp.py                           <- Code to clean CRSP stock data
    │   ├── raw                                   <- Scripts to download raw datasets
    │   │   ├── download.py                       <- Basic class to handle downloading logic
    │   │   ├── chen_zimmermann.py                <- Code to download data from Chen Zimmermann (2020)
    │   │   └── crsp.py                           <- Code to download CRSP stock data
    │   ├── data_pipeline.py                      <- Code for full data processing pipeline
    │   └── useful_files                          <- Paths to useful files used in the data processing pipeline
    │    
    ├── features.py                <- Code to create features for modeling
    │   
    ├── modeling                           <- Scripts to handle ML models
    │   ├── algorithms                     <- folder with machine learning algorithms
    │   │   ├── base_model.py              <- defines Modeler base class
    │   │   ├── elastic_net.py             <- defines ElasticNetModel
    │   │   ├── ensemble.py                <- defines EnsembleModel
    │   │   ├── huber_ols.py               <- defines HuberRegressorModel
    │   │   ├── lstm.py                    <- defines LSTMModel
    │   │   └── neural_networks.py         <- defines FFNNModel
    │   ├── portfolio.py                   <- portfolio building class
    │   ├── rolling_trainer.py             
    │   └── run_experiments.py         
    │   
    ├── support                        <- support files
    │   ├── __init__.py
    │   ├── constants.py               <- common constants used in the project
    │   ├── logger.py                  <- script handling logging logic
    │   ├── wrds_connection.py         <- script handling connection to WRDS server
    │   
    └── plots.py                   <- Code to create visualizations
```

--------

