# Interpretations of Language Models
Replication Code for Paper Fact Recall, Heuristics or Pure Guesswork? Precise Interpretations of Language Models for Fact Completion

## Create the CT Plots

To create the CT results plots (Figure 3 in our paper), we first collect the CT results (TODO: refer to CT section here) and store them under `<CT-results-folder>`, these should correspond to the input to the CT code, stored under `<CT-queries-folder>`. We can then create the plots as follows:

```bash
# Exact recall
# GPT2
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/gpt2-xl/1000_exact.json" \
    --CT_folder "<CT-results-folder>/1000_exact_gpt2_xl" \
    --savefolder "data/create_ct_plots/gpt2-xl/summary_pdfs/exact_recall" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \
# Llama 2 7B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_7B/1000_exact.jsonl" \
    --CT_folder "<CT-results-folder>/1000_exact_llama2_7B" \
    --savefolder "data/create_ct_plots/llama2_7B/summary_pdfs/exact_recall" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
# Llama 2 13B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_13B/1000_exact.jsonl" \
    --CT_folder "<CT-results-folder>/1000_exact_llama2_13B" \
    --savefolder "data/create_ct_plots/llama2_13B/summary_pdfs/exact_recall" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \

# Random guesswork 
# GPT2
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/gpt2-xl/1000_guesswork.json" \
    --CT_folder "<CT-results-folder>/1000_guesswork_gpt2_xl" \
    --savefolder "data/create_ct_plots/gpt2-xl/summary_pdfs/random_guesswork" \
    --filename_template "knowledge_{}_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_7B/1000_guesswork.jsonl" \
    --CT_folder "<CT-results-folder>/1000_guesswork_llama2_7B" \
    --savefolder "data/create_ct_plots/llama2_7B/summary_pdfs/random_guesswork" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \
# Llama 2 13B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_13B/1000_guesswork.jsonl" \
    --CT_folder "<CT-results-folder>/1000_guesswork_llama2_13B" \
    --savefolder "data/create_ct_plots/llama2_13B/summary_pdfs/random_guesswork" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \

# Generic LM
# GPT2
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/gpt2_xl/generic_samples/generic_samples.jsonl" \
    --CT_folder "<CT-results-folder>/generic_samples_gpt2_xl" \
    --savefolder "data/create_ct_plots/gpt2-xl/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_7B/generic_samples/generic_samples.jsonl" \
    --CT_folder "<CT-results-folder>/generic_samples_llama2_7B" \
    --savefolder "data/create_ct_plots/llama2_7B/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \  

# Llama 2 13B
python -m src.create_ct_plots.generate_plots \
    --query_file "<CT-queries-folder>/llama2_13B/generic_samples/generic_samples.jsonl" \
    --CT_folder "<CT-results-folder>/generic_samples_llama2_13B" \
    --savefolder "data/create_ct_plots/llama2_13B/summary_pdfs/generic_samples" \
    --filename_template "{}_candidate_mlp.npz" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \  
```

We also create plots corresponding to combinations of samples and existing CT results (for the combined samples and heuristics recall samples, respectively). For the **combination** dataset that combines the exact recall, biased recall and random guesswork samples equally, we use the following approach:

1. Produce the dataset with the mixed samples using [get_combined_mechanisms_data.ipynb](get_combined_mechanisms_data.ipynb).
2. Get the plots for the dataset using the code below.

```bash
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/gpt2-xl/1000_combined_mechanisms.json" \
    --savefolder "data/create_ct_plots/gpt2-xl/summary_pdfs/combined_mechanisms" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/llama2_7B/1000_combined_mechanisms.json" \
    --savefolder "data/create_ct_plots/llama2_7B/summary_pdfs/combined_mechanisms" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/llama2_13B/1000_combined_mechanisms.json" \
    --savefolder "data/create_ct_plots/llama2_13B/summary_pdfs/combined_mechanisms" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \
```

For the **heuristics recall** samples (that combine prompt bias, person name bias and lexical overlap), we use the following approach to get the CT results plots: 

1. Produce the dataset with the mixed samples using [src/create_ct_plots/get_heuristics_recall_data.ipynb](src/create_ct_plots/get_heuristics_recall_data.ipynb).
2. Get the plots for the dataset using the code below.

```bash
# GPT2
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/gpt2-xl/1000_combined_bias_mechanisms.json" \
    --savefolder "data/create_ct_plots/gpt2-xl/summary_pdfs/heuristics_recall" \
    --arch "gpt2-xl" \
    --archname "GPT-2-XL" \

# Llama 2 7B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/llama2_7B/1000_combined_bias_mechanisms.json" \
    --savefolder "data/create_ct_plots/llama2_7B/summary_pdfs/heuristics_recall" \
    --arch "llama2_7B" \
    --archname "Llama-2-7B" \

# Llama 2 13B
python -m pararel.eval_on_fact_recall_set.generate_plots_for_combined_data \
    --query_file "<CT-queries-folder>/llama2_13B/1000_combined_bias_mechanisms.json" \
    --savefolder "data/create_ct_plots/llama2_13B/summary_pdfs/heuristics_recall" \
    --arch "llama2_13B" \
    --archname "Llama-2-13B" \
```