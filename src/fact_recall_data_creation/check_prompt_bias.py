# takes inspiration from ROME code
# note that GPT-2 XL does not support batch sizes larger than 1 since it does not function with padding.

import argparse
from tqdm import tqdm
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_recall_data_creation.create_paraphrased_queries import PARAPHRASES, parse_prompt
from fact_recall_data_creation.check_name_bias import PersonNameFilter

class PromptBiasFilter(PersonNameFilter):
    person_nouns = ["He", "She"]
    SUBJECTNOUNS = {
        "P17": ["It"],
        "P19": person_nouns,
        "P20": person_nouns,
        "P27": person_nouns,
        "P30": ["It"],
        "P36": ["The country", "The state"],
        "P37": ["The country", "The state"],
        "P39": person_nouns,
        "P101": person_nouns+["The person", "It"], 
        "P103": person_nouns+["The person"],
        "P106": person_nouns,
        "P108": person_nouns,
        "P127": ["It", "The company"],
        "P131": ["It"],
        "P136": ["He", "She", "The band"],
        "P138": ["It"],
        "P140": ["He", "She", "It"],
        "P159": ["It", "The organisation"],
        "P176": ["The product"],
        "P178": ["The software"],
        "P190": ["It"],
        "P276": ["It"],
        "P364": ["The show", "The film"],
        "P407": ["The book"],
        "P413": person_nouns,
        "P449": ["The show"],
        "P463": ["He", "She", "It"],
        "P495": ["It"],
        "P641": person_nouns,
        "P740": ["It", "The organisation"],
        "P937": person_nouns,
        "P1303": person_nouns,
        "P1376": ["It", "The city"],
        "P1412": person_nouns,
    }
    GRAMMARFIXES = {}
    # subjects in the middle should not be capitalized
    for vals in SUBJECTNOUNS.values():
        for val in vals:
            grammarfix_val = " "+val
            if grammarfix_val not in GRAMMARFIXES:
                GRAMMARFIXES[grammarfix_val] = " "+val.lower()
            
    GRAMMARFIXES["He's"] = "His"
    GRAMMARFIXES["he's"] = "his"
    GRAMMARFIXES["She's"] = "Her"
    GRAMMARFIXES["she's"] = "her"

    def __init__(self, top_k, model_name, cache_folder):
        self.do_lower_case = "uncased" in model_name
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token = True
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_folder, use_auth_token = True)
        self.model.to(torch.device("cuda"))
        self.model.eval()
                    
    def filter(self, queries):
        filter_vals = {}
        for (relation, template, pred) in tqdm(queries.groupby(["predicate_id", "used_template", "answers"]).groups.keys()):
            filter_vals[(relation, template, pred)] = self.match(relation, template, pred)
        matches = queries.apply(lambda row: filter_vals[(row.predicate_id, row.used_template, row.answers)], axis=1)
        return matches

    def match(self, relation, template, pred):
        if not relation in self.SUBJECTNOUNS:
            return False

        if self.do_lower_case:
            template = template.lower()
            pred = pred.lower()
            
        for noun in self.SUBJECTNOUNS[relation]:
            prompt = parse_prompt(template, noun)
            # fix e.g. "He's"
            for issue_string, fix in self.GRAMMARFIXES.items():
                prompt = prompt.replace(issue_string, fix)
            
            if pred.strip() in self.get_top_k(prompt):
                return True
        return False
    
    def get_top_k(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        logits = self.model(torch.tensor(input_ids).unsqueeze(0).to(torch.device("cuda")))["logits"][0].detach().cpu()
        
        top_k_ids = torch.topk(logits[-1, :], k=self.top_k)[1].numpy()
        # GPT-2 XL adds space before each token
        top_k_tokens = [self.tokenizer.decode(val).strip() for val in top_k_ids]
        return top_k_tokens

def main(args):
    data = pd.read_json(args.srcfile, lines=True)
      
    print("Processing data for prompt bias...")
    data["used_template"] = data.apply(lambda row: row.prompt.replace(row.sub_label, "[X]"), axis=1)
    pb_filter = PromptBiasFilter(top_k=args.prompt_bias_top_k, model_name=args.model_name, cache_folder=args.cache_folder)
    data["prompt_bias"] = pb_filter.filter(data)

    data.to_json(args.outfile, lines=True, orient="records")
    print("Data with annotations saved!")

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
        "--prompt_bias_top_k",
        default=10,
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
        "--cache_folder",
        default=None,
        help="The cache directory for Huggingface.",
    )
    args = argparser.parse_args()

    print(args)
    main(args)