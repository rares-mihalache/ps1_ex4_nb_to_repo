# Titanic Data Journey

This repository refactors a Jupyter notebook analysing the Titanic dataset into reusable Python modules, following modular programming principles.

## Project structure

- `data/`: the Titanic datasets (`train.csv`, `test.csv`, `gender_submission.csv`)
- `titanic/data.py`: loading and inspecting the data
- `titanic/features.py`: feature engineering
- `titanic/plotting.py`: plotting functions, and `save_fig` for saving figures
- `titanic/model.py`: training and evaluating the model
- `Titanic Data Journey.ipynb`: the analysis notebook, which calls the functions above
- `figures/`: figures saved for the report
- `report.ipynb`: a short report (max 500 words) on which features matter for survival

## Setup

1. Clone the repository
2. Run `pixi install` in the project folder
3. Open the notebook and select the pixi kernel (`.pixi/envs/default`)

## Data

The data comes from the Kaggle Titanic competition.