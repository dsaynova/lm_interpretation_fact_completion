# Interpretations of Language Models

Code for the paper ["Fact Recall, Heuristics or Pure Guesswork? Precise Interpretations of Language Models for Fact Completion"](https://arxiv.org/abs/2410.14405).

## PrISM Dataset Creation
This section describes our method for creating the PrISM datasets. Each PrISM dataset is model-specific. We develop PrISM datasets for GPT-2 XL, Llama2 7B and Llama2 13B. PrISM datasets are created by the identification and generation of samples corresponding to each of the four prediction scenarios _exact fact recall_, _guesswork_, _heuristics recall_, and _generic language modelling_. We describe the method for collecting the samples corresponding to each prediction scenario below.

### 1-2. Exact fact recall and Guesswork

The collection of exact fact recall and guesswork samples, respectively, have many steps in common. Their collection is performed in 9 steps, as follows:
1. Sample fact tuples from the [LAMA dataset](https://github.com/facebookresearch/LAMA).
2. Create queries using [ParaRel](https://github.com/yanaiela/pararel) templates.
3. Collect LM predictions for the queries.
4. Get the average monthly Wikipedia page views for 2019 for each query subject.
5. Get name bias and lexical overlap metadata.
6. Get prompt bias metadata.
7. Label the queries into "correct" and "incorrect". 
8. Add model confidence metadata.
9. Partition the dataset corresponding to exact fact recall and guesswork.

Steps 4-9 can be performed on any input dataset that contains subjects, objects, prompts and model predictions.


#### 1. Sample fact tuples from the LAMA dataset

We sample fact tuples from the [LAMA dataset](https://github.com/facebookresearch/LAMA) corresponding to the relations as listed below:

```python
RELATIONS = {"P19": "place of birth",
             "P20": "place of death",
             "P27": "country of citizenship",
             "P101": "field of work",
             "P495": "country of origin",
             "P740": "location of formation",
             "P1376": "capital of"}
```

The sampling is performed using the scripts as shown below (from the repo root):
```bash
cd data
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
mv data lama_data
```

Collate the data to one dataset using the code below (from the repo root).

```bash
python -m src.fact_recall_data_creation.collate_lama_data \
    --srcdir_trex "data/lama_data/TREx" \
    --srcdir_google_re "data/lama_data/Google_RE" \
    --output_file "data/data_creation/lama_data.jsonl" \
```


#### 2. Create queries using ParaRel templates

Based on the fact tuples sampled from LAMA; we create queries using templates from [ParaRel](https://github.com/yanaiela/pararel) as listed below:

```python
PARAPHRASES = {"P19": ["[X] was born in [Y]",
                       "[X] is originally from [Y]",
                       "[X] was originally from [Y]",
                       "[X] originated from [Y]",
                       "[X] originates from [Y]"
                       ],
               "P20": ["[X] died in [Y]",
                       "[X] died at [Y]",
                       "[X] passed away in [Y]",
                       "[X] passed away at [Y]",
                       "[X] expired at [Y]",
                       "[X] lost their life at [Y]",
                       "[X]'s life ended in [Y]",
                       "[X] succumbed at [Y]"
                       ],
               "P27": ["[X] is a citizen of [Y]",
                       "[X], a citizen of [Y]",
                       "[X], who is a citizen of [Y]",
                       "[X] holds a citizenship of [Y]",
                       "[X] has a citizenship of [Y]",
                       "[X], who holds a citizenship of [Y]",
                       "[X], who has a citizenship of [Y]"
                       ],
               "P101": ["[X] works in the field of [Y]",
                        "[X] specializes in [Y]",
                        "The expertise of [X] is [Y]",
                        "The domain of activity of [X] is [Y]",
                        "The domain of work of [X] is [Y]",
                        "[X]'s area of work is [Y]",
                        "[X]'s domain of work is [Y]",
                        "[X]'s domain of activity is [Y]",
                        "[X]'s expertise is [Y]",
                        "[X] works in the area of [Y]"
                        ],
               "P495": ["[X] was created in [Y]",
                        "[X], that was created in [Y]",
                        "[X], created in [Y]",
                        "[X], that originated in [Y]",
                        "[X] originated in [Y]",
                        "[X] formed in [Y]",
                        "[X] was formed in [Y]",
                        "[X], that was formed in [Y]",
                        "[X] was formulated in [Y]",
                        "[X], formulated in [Y]",
                        "[X], that was formulated in [Y]",
                        "[X] was from [Y]",
                        #"[X], who was from [Y]" subjects are not people
                        "[X], from [Y]",
                        "[X], that was developed in [Y]",
                        "[X] was developed in [Y]",
                        "[X], developed in [Y]"
                        ],
               "P740": ["[X] was founded in [Y]",
                        "[X], founded in [Y]",
                        "[X] that was founded in [Y]",
                        "[X], that was started in [Y]",
                        "[X] started in [Y]",
                        "[X] was started in [Y]",
                        "[X], that was created in [Y]",
                        "[X], created in [Y]",
                        "[X] was created in [Y]",
                        "[X], that originated in [Y]",
                        "[X] originated in [Y]",
                        "[X] formed in [Y]",
                        "[X] was formed in [Y]",
                        "[X], that was formed in [Y]"
                        ],
               "P1376": ["[X] is the capital of [Y]",
                         "[X] is the capital city of [Y]",
                         "[X], the capital of [Y]",
                         "[X], the capital city of [Y]",
                         "[X], that is the capital of [Y]",
                         "[X], that is the capital city of [Y]"
                         ]}
```

We create the queries leveraging the script as shown below:

```bash
python -m src.fact_recall_data_creation.create_paraphrased_queries \
    --srcfile "data/data_creation/lama_data.jsonl" \
    --outfile "data/data_creation/lama_data_queries.jsonl" \
```


#### 3. Collect LM predictions for the queries

We keep all top 10 predictions and store the corresponding model confidences as metadata. This step is LM dependent, as the predictions are unique to each LM we create PrISM dataset for.

Before collecting the model predictions, you may want to download the model weights (to avoid having to do it while using a costly GPU):
pre-load the models to cache. Make sure to use the designated cache folder (`<cache-folder>`) in the subsequent scripts.

```bash
python
```

```python
from src.fact_recall_data_creation.get_model_preds import ModelAndTokenizer
cache_folder = "<cache-folder>"
mt = ModelAndTokenizer(<desired-model>, cache_folder=cache_folder)
```

**GPT-2 XL**

Run the script [scripts/fact_recall_data_creation/get_model_preds/gpt2_xl.sh](scripts/fact_recall_data_creation/get_model_preds/gpt2_xl.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/gpt2_xl/lama_data_preds.jsonl`.

**Llama2 7B**

Run the script [scripts/fact_recall_data_creation/get_model_preds/llama2_7B.sh](scripts/fact_recall_data_creation/get_model_preds/llama2_7B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_7B/lama_data_preds.jsonl`.

**Llama2 13B**

Run the script [scripts/fact_recall_data_creation/get_model_preds/llama2_13B.sh](scripts/fact_recall_data_creation/get_model_preds/llama2_13B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_13B/lama_data_preds.jsonl`.


#### 4. Get the average monthly Wikipedia page views for 2019 for each query subject. 

The Wikipedia page views are added as metadata, no filtering is performed here.

To collect the Wikipedia page views, run the code below:

```bash
python -m src.fact_recall_data_creation.add_wiki_views \
    --srcfile "data/data_creation/gpt2_xl/lama_data_preds.jsonl" \
    --output_file "data/data_creation/gpt2_xl/lama_data_preds_wiki.jsonl" \

python -m src.fact_recall_data_creation.add_wiki_views \
    --srcfile "data/data_creation/llama2_7B/lama_data_preds.jsonl" \
    --output_file "data/data_creation/llama2_7B/lama_data_preds_wiki.jsonl" \

python -m src.fact_recall_data_creation.add_wiki_views \
    --srcfile "data/data_creation/llama2_13B/lama_data_preds.jsonl" \
    --output_file "data/data_creation/llama2_13B/lama_data_preds_wiki.jsonl" \
```

It will collect the views for each of the model-specific datasets.

#### 5. Get name bias and lexical overlap metadata. 

We label tuples following the same approach as [LAMA-UHN](https://github.com/facebookresearch/LAMA/blob/main/scripts/create_lama_uhn.py) using the LM for which the PrISM dataset is created. This step is also LM dependent, as the bias detection depends on the underlying model.

> For all scripts, note that you need to edit them to specify your GPU allocation, set up the virtual enviroment and specify the Hugging Face cache folder for model weights.

**GPT-2 XL**

Run the script [scripts/fact_recall_data_creation/check_name_bias/gpt2_xl.sh](scripts/fact_recall_data_creation/check_name_bias/gpt2_xl.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/gpt2_xl/lama_data_preds_wiki_nb.jsonl`.

**Llama2 7B**

Run the script [scripts/fact_recall_data_creation/check_name_bias/llama2_7B.sh](scripts/fact_recall_data_creation/check_name_bias/llama2_7B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_7B/lama_data_preds_wiki_nb.jsonl`.

**Llama2 13B**

Run the script [scripts/fact_recall_data_creation/check_name_bias/llama2_13B.sh](scripts/fact_recall_data_creation/check_name_bias/llama2_13B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_13B/lama_data_preds_wiki_nb.jsonl`.

#### 6. Get prompt bias metadata

We label queries for which the template without the subject entity yields a correct LM prediction. This approach is inspired by the work on prompt bias by [Cao et al. (2021)](https://aclanthology.org/2021.acl-long.146.pdf). This step is also LM-dependent.

> For all scripts, note that you need to edit them to specify your GPU allocation, set up the virtual enviroment and specify the Hugging Face cache folder for model weights.

**GPT-2 XL**

Run the script [scripts/fact_recall_data_creation/check_prompt_bias/gpt2_xl.sh](scripts/fact_recall_data_creation/check_prompt_bias/gpt2_xl.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/lama_data_preds_wiki_nb_pb.jsonl`.

**Llama2 7B**

Run the script [scripts/fact_recall_data_creation/check_prompt_bias/llama2_7B.sh](scripts/fact_recall_data_creation/check_prompt_bias/llama2_7B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_7B/lama_data_preds_wiki_nb_pb.jsonl`.

**Llama2 13B**

Run the script [scripts/fact_recall_data_creation/check_prompt_bias/llama2_13B.sh](scripts/fact_recall_data_creation/check_prompt_bias/llama2_13B.sh). Make sure that the LM, input file and output file arguments are correct in the script. It will currently save the results to `data/data_creation/llama2_13B/lama_data_preds_wiki_nb_pb.jsonl`.

#### 7-9. Label predictions corresponding to correctness and confidence, and partition the dataset.

We label the queries on whether they are "correct" or "incorrect". We then add metadata on model confidence, proxied by consistency under paraphrasing. The partitioning of the dataset is then performed, and we store samples corresponding to exact fact recall (samples corresponding to no bias with popularity above 1000) and guesswork (predictions that are accurate but inconsistent).

Use the notebook [src/fact_recall_data_creation/final_processing.ipynb](src/fact_recall_data_creation/final_processing.ipynb) for this. Make sure to edit the data sources in the header of the notebook to specify the model of interest (GPT-2 XL, Llama2 7B or Llama2 13B), matching the data that has been generated in steps 1-6.

### 3. Heuristics recall

For heuristic recall we follow these steps:

1. Identify the types of subjects that are allowed or common for each relation. For relations P495 and P740 we query Wikipedia for the types of the LAMA subjects.
2. Use [fantasynamegenerators.com](https://www.fantasynamegenerators.com) to generate the input subjects in folder [synth_subjects](src/heuristics_recall_data_creation/synth_subjects).
3. Perform a check that the generated subjects do not correspond to existing entities in Wikipedia
4. Get model predictions

### 4. Generic language modelling
For generic language samples we follow these steps to extract data from Wikipedia:

1. Download Wikipedia extract
2. For each Wiki page get the first sentence (capped to 10 words) that fulfils the following conditions: longer than 5 words, less than 3 capital letters (likely a heading), next word is not a number or capitalized (likely an entity)
3. Get model prediction
4. Format data

## Mechanistic Interpretations of PrISM scenarios

### Information flow analysis

We perform the analysis proposed by [Geva et al. (2023)](https://arxiv.org/pdf/2304.14767) on the samples from the different prediction scenarios in the PrISM dataset for the GPT-2 XL model. Specifically, we reproduce their information flow analysis (Figure 2 in the paper by Geva et al.) and study of intermediate MHSA and MLP predictions (Figure 5 in the paper by Geva et al.).

We first leverage scripts to collect the information flow results and intermediate predictions for the samples in the PrISM dataset (since this requires the use of a GPU). For this, we run the scripts under scripts/information_flow_analysis. Before this, you may want to create subsets of the PrISM sets such that each set contains 1,000 samples and analyse these. We did this for our analysis.

> Make sure that the PrISM data file paths specified in the scripts are correct.

> For all scripts, note that you need to edit them to specify your GPU allocation, set up the virtual enviroment and potentially specify the Hugging Face cache folder for model weights.

 We then analyse and plot the results in the notebook [src/information_flow_analysis/analyze_information_flow_results.ipynb](src/information_flow_analysis/analyze_information_flow_results.ipynb). This notebook loads the results from the scripts described above and creates the corresponding plots.


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
