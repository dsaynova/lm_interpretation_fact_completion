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

### 4. Generic language modelling
