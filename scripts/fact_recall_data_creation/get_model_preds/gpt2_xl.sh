#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>

python -m src.fact_recall_data_creation.get_model_preds \
    --model_name "gpt2-xl" \
    --srcfile "data/data_creation/lama_data_queries.jsonl" \
    --outfile "data/data_creation/gpt2_xl/lama_data_preds.jsonl" \
    --top_k 3 \