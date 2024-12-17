<div align="center">

# Modelling of the Denitrification Pilot Reactor at Veas

</div>

## Description

This repository contains the code for the paper "Machine learning in wastewater treatment:
insights from modelling a pilot denitrification
reactor". The dataset can be found at [data.sintef.no](https://data.sintef.no/feature/fe-63c812ef-13fe-4b0c-936e-44da3303a781) (public, but requires login). The code base leverages [darts](https://unit8co.github.io/darts/index.html) for model implementations, see their documentation for the models used in this repository and their arguments.

The optimized hyperparameters for each model and task can be found in the [configs/model](configs/model) folder with 
the naming convention veas_pilot_[model]_[task].yaml, e.g. the tcn model for the nowcasting task can be found at [configs/model/veas_pilot_tcn_nowcast.yaml](configs/model/veas_pilot_tcn_nowcast.yaml).

<br>

## How to Run

### Download data

Download the [data](https://data.sintef.no/feature/fe-63c812ef-13fe-4b0c-936e-44da3303a781) and place it in the [data](data) folder, with the name "veas_denit_pilot.csv".

### Install dependencies

```bash
# [OPTIONAL] create conda environment or venv
conda create -n veas python=3.10
conda activate veas

# install pytorch according to instructions.
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# [OPTIONAL] if you cloned the repository to a slurm cluster,
# run the script below to automatically submit jobs through slurm
bash scripts/configure_slurm.sh
```

### Reproduce results

The results of the paper can be replicated by running the script [scripts/run_veas_pilot_forecast_test.sh](scripts/run_veas_pilot_forecast_test.sh) for the forecasting task, and [scripts/run_veas_pilot_nowcast_test.sh](scripts/run_veas_pilot_nowcast_test.sh) for the nowcasting task. 

```bash
bash scripts/run_veas_pilot_forecast_test.sh
```

The results will be saved in the mlflow experiments veas_pilot_forecast_test and veas_pilot_nowcast_test 
respectively. These results can be inspected using mlflow ui, and the plots from the paper can be generated using the
notebooks in the [notebooks](notebooks) folder.

Below we describe how to manually train and evaluate models, and how to run hyperparameter optimization.

### Training

To train models, specify the task using experiment=veas_pilot_nowcast or veas_pilot_forecast, and the model using model=tcn, rnn, xgboost, or elastic_net. For instance, to train a TCN model for the nowcasting task, do:

```bash
python src/train.py experiment=veas_pilot_nowcast model=veas_pilot_tcn_nowcast
```

This will use the first 60/20/20 data split described in the paper, to use the final 72/8/20 split used to generate the results, use the experiment configs with suffix _test, e.g.:

```bash
python src/train.py experiment=veas_pilot_nowcast_test model=veas_pilot_tcn_nowcast
```

This will train the model and save the model itself and the other results to a folder in logs/train/runs/date_time. When evaluating the model later, you will provide this folder to select which model to use. For stochastic models we train multiple models with different seeds to generate the results in the paper, which can be done by setting the seed argument:

```bash
python src/train.py --multirun experiment=veas_pilot_nowcast model=veas_pilot_tcn_nowcast seed='range(10)'
```

See [Hydra documentation](https://hydra.cc/docs/intro/) and the [tutorial document](tutorial.md) for more information.

### Evaluation

The trained model(s) can be evaluated as described in the paper by pointing to a specific model directory (e.g. logs/train/runs/2024-08-28_14-45-35 or logs/train/multiruns/2024-09-06_17-56-21/0), and using the corresponding experiment config for the test dataset:

```bash
python src/eval.py model_dir=logs/train/multiruns/2024-09-06_17-56-21/0 experiment=veas_pilot_nowcast_test
```

### Hyperparameter Optimization

Hyperparameter optimization as described in the paper can be done by making a config file at [configs/hparams_search]
(configs/hparams_search), and using this with the [src/train_cv.py](src/train_cv.py) python script. For example, 
hyperparameter optimization for the xgboost model on the forecasting task can be run with:

```bash
python src/train_cv.py hparams_search=veas_pilot_forecast_cv_xgboost
```

The results are saved in a database at [logs/optuna/hyperopt.db](logs/optuna/hyperopt.db). You can view these results through optuna-dashboard, for instance by running:

```bash
bash scripts/run_optuna_dashboard.sh
```

# Citation

If you use this software in your work, please consider citing:

```latex
@article{rewts,
  title={Recency-Weighted Temporally-Segmented Ensemble for Time-Series Modeling},
  author={Johnsen, P{\aa}l V and B{\o}hn, Eivind and Eidnes, S{\o}lve and Remonato, Filippo and Riemer-S{\o}rensen, Signe},
  journal={arXiv preprint arXiv:2403.02150},
  year={2024}
}
```
