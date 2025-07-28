import argparse
import json
import os
from collections import defaultdict, Counter
import requests
import time
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_topX_token(model, tokenizer, prompts, X, return_p=False, tok_id=False):
    inp = make_inputs(tokenizer, prompts)
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.topk(probs, X, dim=1)
    # print(preds)
    result = [tokenizer.decode(c) for c in preds[0]]
    if return_p:
        if tok_id:
            return result, p[0].tolist(), preds[0]
        return result, p[0].tolist()
    if tok_id:
        return result, preds[0]
    return result


def make_inputs(tokenizer, prompts, device="cuda", add_special_tokens=True):
    if "LlamaTokenizer" in str(type(tokenizer)):
        token_lists = [tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in prompts]
    else:
        token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    assert all([len(token_list) == maxlen for token_list in
                token_lists]), "Inputs must be of same length, GPT2-XL does not support padding"
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def read_jsonl_file(filename: str):
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)
    return dataset


def wiki_check(rel, option):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?item ?value ?valueLabel
    {
      ?item wdt:'''
    query2 = ''' ?value.
      ?value ?label "'''
    query3 = '''"@en .

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    }
    LIMIT 1
    '''
    query = query1 + rel + query2 + option.strip() + query3
    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1:
        return True
    else:
        return False


def wiki_subject_check(rel, sub):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?item ?value ?valueLabel
    {
      ?item wdt:'''
    query2 = ''' ?value.
      ?value ?label "'''
    query3 = '''"@en .

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    }
    LIMIT 1
    '''
    query = query1 + rel + query2 + option.strip() + query3
    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1:
        return True
    else:
        return False


def wiki_label_from_Q(q):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?valueLabel
    WHERE
    {
    wd:'''
    query2 = ''' wdt:P31 ?value.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    } 
    LIMIT 1  '''
    query = query1 + q + query2
    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1:
        return data['results']['bindings'][0]['valueLabel']['value']
    else:
        return "not found"

relations = ["P19", "P20", "P27", "P101", "P1376", "P740", "P495"]

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

SUBJECTS = {"P19": ["DND_human_male",
                    "DND_human_female",
                    "Russian",
                    "French",
                    "German",
                    "Korean",
                    "Japanese"
                   ],
            "P20": ["DND_human_male",
                    "DND_human_female",
                    "Russian",
                    "French",
                    "German",
                    "Korean",
                    "Japanese"
                   ],
            "P27": ["DND_human_male",
                    "DND_human_female",
                    "Russian",
                    "French",
                    "German",
                    "Korean",
                    "Japanese"
                   ],
            "P101": ["DND_human_male",
                     "DND_human_female",
                     "Russian",
                     "French",
                     "German",
                     "Korean",
                     "Japanese"
                    ],
            "P495": ["Music_group",
                     "AnimeAndManga",
                     "Books",
                     "Newspapers",
                     "Magazines"
                    ],
            "P740": ["Music_group",
                     "Company"
                    ],
            "P1376": ["Town_Central_Africa",
                      "Town_Central_America",
                      "Town_Central_Asia",
                      "Town_East_Asia",
                      "Town_East_Europe",
                      "Town_Middle_Eastern",
                      "Town_West_Europe"
                     ]}

SUBJECTNOUNS = {
    "P19": ["He", "She"],
    "P20": ["He", "She"],
    "P27": ["He", "She"],
    "P101": ["He", "She"],
    "P495": ["It"],
    "P740": ["It", "The organisation"],
    "P1376": ["It", "The city"]
}
GRAMMARFIXES = {"He's": "His",
                "She's": "Her"
                }

def main(model_name, cache_folder, data_folder, lama_folder, results_folder):

    torch_dtype = torch.float16 if "20b" in model_name else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype,use_auth_token=True, cache_dir=cache_folder)
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True,cache_dir=cache_folder)
    #GET VALID OPTIONS
    for relation in relations[4:]:
        options = set()

        for source in SUBJECTS[relation]:
            subjects_all = []

            with open(f"{data_folder}/{source}.pickle", "rb") as f:
                subjects_all = pickle.load(f)

            for n in subjects_all:

                for template in PARAPHRASES[relation]:
                    example = template.replace("[X]", n).replace(" [Y]", "")
                    tokens, probs = predict_topX_token(model, tokenizer, [example], 3, return_p=True)
                    options.update(tokens)
        possible_options = []
        trivial_options = []

        for p in options:

            if relation == "P101" and len(p.strip()) > 2 and wiki_check(relation, p):
                possible_options.append(p.strip())

            elif relation != "P101" and len(p.strip()) > 0 and p.strip()[0].isupper():
                possible_options.append(p.strip())

            else:
                trivial_options.append(p.strip())

        if relation == "P101":
            lama_options = set()
            data = read_jsonl_file(os.path.join(lama_folder, relation + ".jsonl"))

            for dp in data:
                lama_options.update({dp["obj_label"]})

            # for Llama2-7B - add LAMA options, since Llama tokenizer is too strict
            possible_options_from_lama = list(
                set([x for x in trivial_options for y in lama_options if len(x) > 2 and x in y and y.index(x) == 0]))
            possible_options.extend(possible_options_from_lama)

            possible_options.extend(["automotive", "financial", "electronic", "investigative", "medicinal",
                                     "mechanical", "historical", "digital", "autonomous", "economic",
                                     "artificial", "legal", "IT", "agricultural", "sustainable", "political",
                                     "computational", "nuclear", "forensic", "neuro", "cyber", "molecular",
                                     "veterinary", "environmental", "cultural", "quantum",
                                     # Llama2-7B
                                     "agricult", "crypt", "sculpt", "entertain", "dent", "mathematical", "manufact"])
            possible_options.remove("the")
            possible_options.remove("for")
            possible_options.remove("his")

        with open(f'{results_folder}/valid_options/{relation}_options.pickle', 'wb') as outputfile:
            pickle.dump(possible_options, outputfile)
        with open(f'{results_folder}/valid_options/{relation}_trivial.pickle', 'wb') as outputfile:
            pickle.dump(trivial_options, outputfile)

    #MODEL PREDICTION

    #Prompt bias
    top10_per_rel_noun = {}
    for rel, templates in PARAPHRASES.items():
        top10_per_rel_noun[rel] = {}
        for template in templates:
            top10_per_rel_noun[rel][template] = {}
            for noun in SUBJECTNOUNS[rel]:
                prompt = template.replace("[X]", noun).replace(" [Y]", "")
                # fix e.g. "He's"
                for issue_string, fix in GRAMMARFIXES.items():
                    prompt = prompt.replace(issue_string, fix)

                tokens, probs = predict_topX_token(model, tokenizer, [prompt], 10, return_p=True)
                top10_per_rel_noun[rel][template][noun] = tokens

    #Prediction
    all_synth_data = []
    for relation in relations:
        print(relation)

        with open(f'{results_folder}/valid_options/{relation}_options_no_month.pickle', 'rb') as outputfile:
            possible_options = pickle.load(outputfile)

        for source in SUBJECTS[relation]:
            subjects_all = []

            with open(f"{data_folder}/{source}.pickle", "rb") as f:
                subjects_all = pickle.load(f)

            for n in subjects_all:
                confidence = Counter()

                for template in PARAPHRASES[relation]:
                    example = template.replace("[X]", n).replace(" [Y]", "")
                    tokens, probs = predict_topX_token(model, tokenizer, [example], 3, return_p=True)
                    valid = [tokens.index(t) for t in tokens if t.strip() in possible_options]
                    trivial = []
                    non_trivial = []

                    if len(valid) == 0:
                        trivial = tokens

                    elif len(valid) < 3:
                        non_trivial = [tokens[i] for i in valid]
                        trivial = [i for i in tokens if i not in non_trivial]

                    else:
                        non_trivial = tokens
                    confidence.update(tokens)
                    result_dict = defaultdict()
                    result_dict["predicate_id"] = relation
                    result_dict["subject"] = n
                    result_dict["template"] = template
                    result_dict["prompt"] = example
                    result_dict["answers"] = tokens
                    result_dict["p_answers"] = probs
                    result_dict["non_trivial"] = non_trivial
                    result_dict["trivial"] = trivial
                    result_dict["source"] = source

                    for noun, preds in top10_per_rel_noun[relation][template].items():
                        result_dict[f"answers_for_PB_{noun.replace(' ', '_')}"] = preds
                    all_synth_data.append(result_dict)

                for record in all_synth_data:

                    if record["subject"] == n and record["predicate_id"] == relation: record[
                        "confidence"] = confidence

        with open(f'{results_folder}/all_synth_data_model_labels.pickle', 'wb') as outputfile:
            pickle.dump(all_synth_data, outputfile)
        print(len(all_synth_data))

    #Get top non-trivial prediction
    output_file = (f"{results_folder}/all_synth_data_top_non_trivial.jsonl")
    outfile = open(output_file, "w")

    for i, record in enumerate(all_synth_data):

        if len(record["non_trivial"]) == 0: continue
        record_copy = record.copy()
        record_copy["candidate_prediction"] = record["non_trivial"][0]
        record_copy["candidate_p"] = record["p_answers"][record["answers"].index(record["non_trivial"][0])]
        record_copy["known_id"] = i
        record_copy["top10_tokens"] = record["answers"]
        record_copy["prediction_p"] = record["p_answers"]
        record_copy["prediction"] = record["answers"][0]
        record_copy["confident_flag"] = record["confidence"][record["non_trivial"][0]] > 4
        record_copy["prompt_bias"] = False

        for noun in top10_per_rel_noun[record["predicate_id"]][record["template"]].keys():
            label = f"answers_for_PB_{noun.replace(' ', '_')}"

            if record["non_trivial"][0] in record[label]:
                record_copy["prompt_bias"] = True
        outfile.write(json.dumps(record_copy))
        outfile.write("\n")
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True)
    parser.add_argument(
        "--cache_folder", required=True)
    parser.add_argument(
        "--data_folder", required=True)
    parser.add_argument(
        "--lama_folder", required=True)
    parser.add_argument(
        "--results_folder", required=True)


    args = parser.parse_args()

    main(
        args.model_name,
        args.cache_folder,
        args.data_folder,
        args.lama_folder,
        args.results_folder
    )