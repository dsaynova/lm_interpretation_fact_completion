import argparse
from datasets import load_dataset
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def predict_topX_token(model, tokenizer, prompts, X, return_p=False):
    inp = make_inputs(tokenizer, prompts)
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.topk(probs, X, dim=1)
    result = [tokenizer.decode(c) for c in preds[0]]
    if return_p:
        return result, p[0].tolist()
    return result

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    assert all([len(token_list)==maxlen for token_list in token_lists]), "Inputs must be of same length, GPT2-XL does not support padding"
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

def main(model_name, cache_folder, results_folder):
    torch_dtype = torch.float16 if "20b" in model_name else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype,
        use_auth_token=True, cache_dir=cache_folder, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True,
                                              cache_dir=cache_folder, device_map="auto")
    data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_folder)

    output_file = (f"{results_folder}/generic_samples.jsonl")
    checked_index = (f"{results_folder}/checked_index.txt")

    random.seed(42)

    outfile = open(output_file, "w")
    index_file = open(checked_index, "w")

    generic_exmples = 0
    checked = set()
    while generic_exmples < 1000:
        ind = random.randint(0, data.num_rows - 1)
        if ind in checked: continue

        checked.add(ind)
        index_file.write(str(ind))
        index_file.write("\n")

        sample = data[ind]
        selected = False
        for i in sample["text"].split(".")[1:]:
            if selected: continue
            for j in sample["title"].split():
                if selected: continue
                if len(j) > 3 and j in i.split() and i.split().index(j) == 0:

                    example = " ".join(i.split()[:10])

                    # filter out too short sentences
                    if len(i.split()) < 5: continue

                    # filter out likely section titles
                    cap = 0
                    for letter in example:
                        if letter.isupper(): cap += 1
                    if cap > 3: continue

                    # filter out likely formatting issue
                    if "  " in example: continue

                    # filter out likely to prompt for fact (if proper name or number)
                    if not example.split()[-1][0].isalpha() or not example.split()[-1][0].islower(): continue

                    span = 0
                    for ind_, tok in enumerate(example.split()):
                        if tok.lower() not in sample["title"].lower().split():
                            span = ind_
                            break

                    prediction, prob = predict_topX_token(model, tokenizer, [" ".join(example.split()[:-1])], 1,
                                                          return_p=True)
                    record = {"known_id": generic_exmples,
                              "original_id": ind,
                              "prompt": " ".join(example.split()[:-1]),
                              "title": sample["title"],
                              "subject": " ".join(example.split()[0:int(span)]),
                              "candidate_prediction": prediction[0],
                              "probability": prob[0],
                              "true": example.split()[-1]
                              }
                    generic_exmples += 1
                    outfile.write(json.dumps(record))
                    outfile.write("\n")
                    selected = True
    outfile.close()
    index_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True)
    parser.add_argument(
        "--cache_folder", required=True)
    parser.add_argument(
        "--results_folder", required=True)


    args = parser.parse_args()

    main(
        args.model_name,
        args.cache_folder,
        args.results_folder
    )