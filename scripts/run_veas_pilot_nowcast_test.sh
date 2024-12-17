#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"/.. || exit
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

stochastic_models=("veas_pilot_xgboost_nowcast" "veas_pilot_tcn_nowcast" "veas_pilot_rnn_nowcast")
models=("veas_pilot_elastic_net_nowcast")
baselines=("veas_pilot_nowcast_test_baseline_running_mean" "veas_pilot_nowcast_test_baseline_train_mean")

# Convert the array to a comma-separated string
IFS=','       # Set the Internal Field Separator to a comma
all_models=$(printf "%s," "${models[@]}")  # Use printf to join array elements with a comma
all_models=${all_models%,}  # Remove the trailing comma

all_stochastic_models=$(printf "%s," "${stochastic_models[@]}")  # Use printf to join array elements with a comma
all_stochastic_models=${all_stochastic_models%,}  # Remove the trailing comma

all_baselines=$(printf "%s," "${baselines[@]}")  # Use printf to join array elements with a comma
all_baselines=${all_baselines%,}  # Remove the trailing comma

echo "Training and evaluating models"
echo "python src/train.py -m experiment=veas_pilot_nowcast_test model=${all_models}"
python src/train.py -m experiment=veas_pilot_nowcast_test model="${all_models}"

echo "python src/train.py -m experiment=veas_pilot_nowcast_test model=${all_stochastic_models}"
python src/train.py -m experiment=veas_pilot_nowcast_test model="${all_stochastic_models}" seed='range(10)'

echo "Evaluating baselines"
echo "python src/train.py -m experiment=${all_baselines}"
python src/train.py -m experiment="${all_baselines}"
