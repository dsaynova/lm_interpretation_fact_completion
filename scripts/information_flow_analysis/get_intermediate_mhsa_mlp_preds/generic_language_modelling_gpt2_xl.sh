#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>

MODEL_NAME="gpt2-xl"
python -m src.information_flow_analysis.get_intermediate_mhsa_mlp_preds \
    --data_path "data/data_creation/${MODEL_NAME}/generic_samples/generic_samples.jsonl" \
    --save_folder "data/information_flow_analysis" \
    --model_name $MODEL_NAME
