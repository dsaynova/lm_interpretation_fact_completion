#!/usr/bin/env bash

set -eo pipefail

# <set up the environment>

python -m src.fact_recall_data_creation.check_prompt_bias \
    --srcfile "data/data_creation/gpt2_xl/lama_data_preds_wiki_nb.jsonl" \
    --outfile "data/data_creation/gpt2_xl/lama_data_preds_wiki_nb_pb.jsonl" \
    --model_name "gpt2-xl" \
    --prompt_bias_top_k 10 \