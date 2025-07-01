#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>
# make sure to have set your cache folder for Huggingface (model weights will be stored here)

python -m src.fact_recall_data_creation.get_model_preds \
    --model_name "meta-llama/Llama-2-13b-hf" \
    --srcfile "data/data_creation/lama_data_queries.jsonl" \
    --outfile "data/data_creation/llama2_13B/lama_data_preds.jsonl" \
    --top_k 3 \
    --cache_folder "<cache-folder>" \