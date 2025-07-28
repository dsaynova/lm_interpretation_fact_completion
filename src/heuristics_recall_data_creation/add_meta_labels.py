import argparse
import torch
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import requests
import time
import random

from transformers import AutoModelForCausalLM, AutoTokenizer


def wiki_lang_to_country(lang):
    url = 'https://query.wikidata.org/sparql'
    query1 = '''
    SELECT ?item ?value ?valueLabel
    {
      ?item wdt:P17 ?value.
      ?item wdt:P31/wdt:P279* wd:Q34770.
      ?item ?label "'''
    query2 = '''"@en .

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
    }
    LIMIT 1
    '''
    query = query1 + lang.strip() + query2
    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    time.sleep(1)
    if len(data['results']['bindings']) == 1:
        return data['results']['bindings'][0]["valueLabel"]["value"]
    else:
        return False


class LAMAUHNFilter:
    def match(self, sub_label, obj_label, relation):
        raise NotImplementedError()

    def filter(self, queries):
        return [self.match(query) for _, query in tqdm(queries.iterrows(), total=len(queries), position=0, leave=True)]


class PersonNameFilter(LAMAUHNFilter):
    TEMP = "[X] is a common name in the following [Y]:"

    PLACENOUNS = {
        "P19": ["city", "country"],
        "P20": ["city", "country"],
        "P27": ["city", "country"],
        # "P1376": ["language"],
        "P1412": ["language"],
        "P103": ["language"],
    }

    def __init__(self, top_k, model_name, cache):
        super().__init__()
        self.do_lower_case = "uncased" in model_name
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache)
        self.model.to(torch.device("cuda"))
        self.model.eval()

    def get_top_k_for_name(self, template, name):
        tokens = self.tokenizer.tokenize(template.replace("[X]", name))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        logits = self.model(torch.tensor(input_ids).unsqueeze(0).to(torch.device("cuda")))["logits"][0].detach().cpu()

        top_k_ids = torch.topk(logits[-1, :], k=self.top_k)[1].numpy()
        # GPT-2 XL adds space before each token
        top_k_tokens = [self.tokenizer.decode(val).strip() for val in top_k_ids]
        return top_k_tokens

    def match(self, query):
        relation = query["predicate_id"]
        if not relation in self.PLACENOUNS:
            return False
        if len(query["non_trivial"]) == 0: return False

        sub_label, obj_label = query["subject"], query["non_trivial"][0]
        if self.do_lower_case:
            obj_label = obj_label.lower()
            sub_label = sub_label.lower()

        # the GPT-2 XL tokenizer may split some objects into multiple tokens
        first_obj_token = self.tokenizer.convert_tokens_to_string(
            [self.tokenizer.tokenize(obj_label, add_special_tokens=False)[0]])
        for option in self.PLACENOUNS[relation]:
            template = self.TEMP.replace("[Y]", option)
            for name in sub_label.split():
                if first_obj_token.strip() in self.get_top_k_for_name(template, name):
                    return True
        return False


class StringMatchFilter(LAMAUHNFilter):
    def __init__(self, do_lower_case):
        self.do_lower_case = do_lower_case

    def match(self, query):
        if len(query["non_trivial"]) == 0: return False

        sub_label, obj_label = query["subject"], query["non_trivial"][0].strip()
        if self.do_lower_case:
            sub_label = sub_label.lower()
            obj_label = obj_label.lower()
        return obj_label in sub_label and len(obj_label) > 2


def read_jsonl_file(filename: str):
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)
    return dataset



def main(model_name, cache, data_dir):
    srcfile = f"{data_dir}/all_synth_data_top_non_trivial.jsonl"
    outfile = f"{data_dir}/all_synth_data_top_non_trivial_all_flags.jsonl"
    filters = ["string_match", "person_name"]#, "person_name")
    string_match_do_lowercase = True
    person_name_top_k = 10

    data = pd.read_json(srcfile, lines=True)

    uhn_filters = []
    if "string_match" in filters:
        uhn_filters.append(
            StringMatchFilter(do_lower_case=string_match_do_lowercase)
        )
    if "person_name" in filters:
        uhn_filters.append(
            PersonNameFilter(
                model_name=model_name, top_k=person_name_top_k, cache=cache
            )
        )

    for ix, uhn_filter in enumerate(uhn_filters):
        print(f"Processing data for filter {filters[ix]}...")
        data[filters[ix]] = uhn_filter.filter(data)

    data.to_json(outfile, lines=True, orient="records")
    print("Data with filter annotations saved!")


    #GET CT EXTRACT

    data_labels = read_jsonl_file(outfile)
    data_labels_id = defaultdict()
    for record in data_labels:
        data_labels_id[record["known_id"]] = record

    data_labels_pd = pd.DataFrame(data_labels)

    random.seed(42)
    random_sample = defaultdict()
    for field in ["prompt_bias", "string_match", "person_name", "no"]:
        a = data_labels_pd["confident_flag"] == True
        b = data_labels_pd["prompt_bias"] == (field == "prompt_bias")
        c = data_labels_pd["string_match"] == (field == "string_match")
        d = data_labels_pd["person_name"] == (field == "person_name")
        valid = data_labels_pd[(a) & (b) & (c) & (d)]["known_id"].tolist()
        sample_size = min(len(valid), 1000)
        random_sample[field] = random.sample(valid, sample_size)

    for i, v in random_sample.items():
        outfile = open(f"{data_dir}/{i}_bias.jsonl", "w")
        for r_id in v:
            outfile.write(json.dumps(data_labels_id[r_id]))
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


    args = parser.parse_args()

    main(
        args.model_name,
        args.cache_folder,
        args.data_folder,
    )

