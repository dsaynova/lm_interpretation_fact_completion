#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>
# make sure to have set your cache folder for Huggingface (model weights will be stored here)

python -m src.fact_recall_data_creation.check_prompt_bias \
    --srcfile "data/data_creation/llama2_7B/lama_data_preds_wiki_nb.jsonl" \
    --outfile "data/data_creation/llama2_7B/lama_data_preds_wiki_nb_pb.jsonl" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --prompt_bias_top_k 10 \
    --cache_folder "<cache-folder>" \