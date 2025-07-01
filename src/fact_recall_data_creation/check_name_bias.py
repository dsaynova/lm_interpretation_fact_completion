# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Code to create LAMA-UHN, a subset of LAMA-Google-RE and LAMA-T-REx
# where ``easy-to-guess'' questions are filtered out.
#
# Defaults parameters correspond to setup in the following paper:
#
# @article{poerner2019bert,
#  title={BERT is Not a Knowledge Base (Yet): Factual Knowledge vs.
#    Name-Based Reasoning in Unsupervised QA},
#  author={Poerner, Nina and Waltinger, Ulli and Sch{\"u}tze, Hinrich},
#  journal={arXiv preprint arXiv:1911.03681},
#  year={2019}
# }

import torch
import argparse
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

class LAMAUHNFilter:
    def match(self, relation, sub_label, obj_label):
        raise NotImplementedError()
    
    def filter(self, queries):
        filter_vals = {}
        
        for (relation, sub_label, pred) in tqdm(queries.groupby(["predicate_id", "sub_label", "answers"]).groups.keys()):
            filter_vals[(relation, sub_label, pred)] = self.match(relation, sub_label, pred)
        matches = queries.apply(lambda row: filter_vals[(row.predicate_id, row.sub_label, row.answers)], axis=1)
        return matches


class PersonNameFilter(LAMAUHNFilter):
    TEMP = "[X] is a common name in the following [Y]:"
    
    # could add occupation? (P106, P101)
    # sport? P641
    PLACENOUNS = {
        "P19": ["city", "country"],
        "P20": ["city", "country"],
        "P27": ["city", "country"],
        "P1412": "language", # not in our dataset (but in counterfact)
        "P103": "language", # not in our dataset (but in counterfact)
    }

    def __init__(self, top_k, model_name, cache_folder):
        super().__init__()
        self.do_lower_case = "uncased" in model_name
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token = True
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token = True, cache_dir=cache_folder)
        self.model.to(torch.device("cuda"))
        self.model.eval()
        
        self.model_is_llama = "llama" in model_name.lower()

    def get_top_k_for_name(self, template, name):
        tokens = self.tokenizer.tokenize(template.replace("[X]", name))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        logits = self.model(torch.tensor(input_ids).unsqueeze(0).to(torch.device("cuda")))["logits"][0].detach().cpu()
        
        top_k_ids = torch.topk(logits[-1, :], k=self.top_k)[1].numpy()
        # GPT-2 XL adds space before each token
        top_k_tokens = [self.tokenizer.decode(val).strip() for val in top_k_ids]
        return top_k_tokens

    def match(self, relation, sub_label, obj_label):
        if not relation in self.PLACENOUNS:
            return False

        if self.do_lower_case:
            obj_label = obj_label.lower()
            sub_label = sub_label.lower()

        if self.model_is_llama:
            # the first token is always the BOS <s> token, skip this
            # also, the model may output "" (potentially starting a number prediction)
            if obj_label == "":
                first_obj_token = ""
            else:
                first_obj_token = self.tokenizer.decode(self.tokenizer.encode(obj_label, add_special_tokens=False)[0])
        else:
            # the GPT-2 XL tokenizer may split some objects into multiple tokens, and requires a space before tokens
            first_obj_token = self.tokenizer.tokenize(" "+obj_label.strip())[0].strip("Ġ")
        for placenoun in self.PLACENOUNS[relation]:
            template = self.TEMP.replace("[Y]", placenoun)
            for name in sub_label.split():
                if first_obj_token in self.get_top_k_for_name(template, name):
                    return True
        return False


class StringMatchFilter(LAMAUHNFilter):
    def __init__(self, do_lower_case):
        self.do_lower_case = do_lower_case
    
    def match(self, _, sub_label, obj_label):
        if self.do_lower_case:
            sub_label = sub_label.lower()
            obj_label = obj_label.lower()
        return obj_label.strip() in sub_label


def main(args):
    tmp_data = pd.read_json(args.srcfile, lines=True)
    # we won't need the columns below and they obstruct the data processing
    tmp_data = tmp_data.drop(columns=["evidences", "obj_aliases"], errors="ignore")
    
    # expand the preds across rows
    tmp_data["rank_answers"] = [list(range(len(tmp_data.iloc[0].answers)))]*len(tmp_data)
    cols_to_explode = ["answers", "p_answers", "rank_answers"]
    non_explode_cols = list(tmp_data.columns)
    for col in cols_to_explode:
        non_explode_cols.remove(col)
    data = tmp_data.set_index(non_explode_cols).apply(pd.Series.explode).reset_index()

    uhn_filters = []
    if "string_match" in args.filters:
        uhn_filters.append(
            StringMatchFilter(do_lower_case=args.string_match_do_lowercase)
        )
    if "person_name" in args.filters:
        uhn_filters.append(
            PersonNameFilter(
                model_name=args.model_name, top_k=args.person_name_top_k, cache_folder=args.cache_folder
            )
        )
      
    for ix, uhn_filter in enumerate(uhn_filters):
        print(f"Processing data for filter {args.filters[ix]}...")
        data[args.filters[ix]] = uhn_filter.filter(data)

    data.to_json(args.outfile, lines=True, orient="records")
    print("Data with filter annotations saved!")
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--srcfile",
        required=True,
        type=str,
        help="Source jsonl file.",
    )
    argparser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="File to save filtered data to.",
    )
    argparser.add_argument(
        "--filters",
        nargs="+",
        type=str,
        default=("string_match", "person_name"),
        choices=("string_match", "person_name"),
        help="Filters to be applied: string_match, person_name or both.",
    )
    argparser.add_argument(
        "--person_name_top_k",
        default=3,
        type=int,
        help="Parameter k for person name filter.",
    )
    argparser.add_argument(
        "--model_name",
        default="gpt2-xl",
        type=str,
        help="ARM to use for the person name filter.",
    )
    argparser.add_argument(
        "--no_string_match_do_lowercase",
        default=True,
        action="store_false",
        dest="string_match_do_lowercase",
        help="Set flag to disable lowercasing in string match filter",
    )
    argparser.add_argument(
        "--cache_folder",
        default=None,
        help="The cache directory for Huggingface.",
    )
    args = argparser.parse_args()

    print(args)
    main(args)
