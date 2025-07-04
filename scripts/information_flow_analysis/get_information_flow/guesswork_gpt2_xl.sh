#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>

MODEL_NAME="gpt2-xl"
python -m src.information_flow_analysis.get_information_flow \
    --data_path "data/data_creation/${MODEL_NAME}/random_guesswork_set.json" \
    --save_folder "data/information_flow_analysis" \
    --model_name $MODEL_NAME
