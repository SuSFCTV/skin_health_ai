# Skin Health AI
Searching for skin lesions

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for skin_health_ai
│                         and configuration for tools like black
│
├── .pre-commit-config <- Config for git hook - scripts are useful for identifying simple issues before submission to code review
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── telegram_bot       <- Source code for telegram bot for using ml models
│
└── skin_health_ai     <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes skin_health_ai a Python module
    │
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── train_utils    <- utils for training (train_loop, losses, etc)
    │   ├── segnet         <- segnet model
    │   └── unet           <- unet model
    │
    └── metrics  <- Scripts for getting metrics
        └── iou.py
```

--------
