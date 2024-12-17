#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"/.. || exit
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

nowcast_regression="python src/train.py experiment=veas_pilot_lags_regression_nowcast model=veas_pilot_xgboost_nowcast,veas_pilot_elastic_net_nowcast"
forecast_regression="python src/train.py experiment=veas_pilot_lags_regression_forecast model=veas_pilot_xgboost_forecast,veas_pilot_elastic_net_forecast"

nowcast_torch="python src/train.py experiment=veas_pilot_lags_torch_nowcast model=veas_pilot_rnn_nowcast,veas_pilot_tcn_nowcast"
forecast_torch="python src/train.py experiment=veas_pilot_lags_torch_forecast model=veas_pilot_rnn_forecast,veas_pilot_tcn_forecast"

echo "Training regression nowcast":
echo "$nowcast_regression"
$nowcast_regression &
$nowcast_regression ++model_lags=1 &

echo "Training regression forecast":
echo "$forecast_regression"
$forecast_regression &
$forecast_regression ++model_lags=1 &

echo "Training torch nowcast":
echo "$nowcast_torch"
$nowcast_torch &
$nowcast_torch ++model_lags=1 &

echo "Training torch forecast":
echo "$forecast_torch"
$forecast_torch &
$forecast_torch ++model_lags=1 &
