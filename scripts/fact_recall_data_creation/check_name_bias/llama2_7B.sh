#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>
# make sure to have set your cache folder for Huggingface (model weights will be stored here)

python -m src.fact_recall_data_creation.check_name_bias \
    --srcfile "data/data_creation/llama2_7B/lama_data_preds_wiki.jsonl" \
    --outfile "data/data_creation/llama2_7B/lama_data_preds_wiki_nb.jsonl" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --person_name_top_k 10 \
    --cache_folder "<cache-folder>" \