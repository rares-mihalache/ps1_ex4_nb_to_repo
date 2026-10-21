# Titanic Data Journey

This repository refactors a Jupyter notebook analysing the Titanic dataset into reusable Python modules, following modular programming principles.

## Project structure

- `data/`: the Titanic datasets (`train.csv`, `test.csv`, `gender_submission.csv`)
- `titanic/data.py`: loading and inspecting the data
- `titanic/features.py`: feature engineering
- `titanic/plotting.py`: plotting functions
- `titanic/model.py`: training and evaluating the model
- `Titanic Data Journey.ipynb`: the analysis notebook, which calls the functions above

## Setup

1. Clone the repository
2. Run `pixi install` in the project folder
3. Open the notebook and select the pixi kernel (`.pixi/envs/default`)

## Data

The data comes from the Kaggle Titanic competition.