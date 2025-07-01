# utilities
from src.information_flow_analysis.utils import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    set_act_get_hooks,
    remove_hooks
)

import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

torch.set_grad_enabled(False)
tqdm.pandas()

def get_intermediate_mhsa_mlp_preds(data, model_name, cache_dir):
    # also collects attribute extraction rates across layers for the MHSA and MLP sublayers
    # i.e. when information copied to the last token state matched the final prediction
    # matches figure 5 in the Geva et al. paper
    print(f"Loading {model_name}...")
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        cache_dir=cache_dir
    )
    mt.model.eval()
    
    E = mt.model.get_input_embeddings().weight
    k = 10

    records = []
    for row_i, row in tqdm(data.iterrows(), total=len(data), desc="Collecting intermediate MHSA and MLP preds"):
        prompt = row.prompt
        subject = row.subject
        gold_attribute = row.attribute if 'attribute' in row.index else None
        
        inp = make_inputs(mt.tokenizer, [prompt])
        input_tokens = decode_tokens(mt.tokenizer, inp["input_ids"][0])
        try:
            e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
            e_range = [x for x in range(e_range[0], e_range[1])]
        except ValueError as e:
            print(f"Encountered a problem processing entity range for sample {row_i}:")
            print(e)
            print("skipping...")
            continue
        non_e_range_last = [x for x in range(len(input_tokens)-1) if x not in e_range]
        source_index = len(input_tokens) - 1
        
        # set hooks to get ATTN and MLP outputs
        hooks = set_act_get_hooks(mt.model, source_index, mlp=True, attn_out=True)
        output = mt.model(**inp)
        # remove hooks
        remove_hooks(hooks)
        
        # for the PRISM datasets, we wish to track a certain model prediction
        # does not have to be top ranked
        # identify it by looking for a col that only exists in PRISM
        if "rank_answers" in row.index:
            attribute_tok = mt.tokenizer.encode(row.prediction, add_special_tokens=False)
            assert len(attribute_tok) == 1
            attribute_tok = attribute_tok[0]
            attribute_tok_final_rank = row.rank_answers
        else:
            # for the knowns dataset, we need to get the model prediction
            probs = torch.softmax(output["logits"][:, -1], dim=1)
            _, attribute_tok = torch.max(probs, dim=1)
            attribute_tok = attribute_tok.cpu().item()
            attribute_tok_final_rank = 0
        [attribute_tok_str] = decode_tokens(mt.tokenizer, [attribute_tok])
        
        for layer in range(mt.num_layers):
            # ATTN
            attn_out = mt.model.activations_[f'attn_out_{layer}'][0]
            proj = attn_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    "example_index": row_i,
                    "prompt": prompt,
                    "subject": subject,
                    "gold_attribute": gold_attribute,
                    "attribute_tok": attribute_tok,
                    "attribute_tok_str": attribute_tok_str,
                    "attribute_tok_final_rank": attribute_tok_final_rank,
                    "layer": layer,
                    "proj_vec": "MHSA",
                    "top_k_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                })
            
            # MLP
            mlp_out = mt.model.activations_[f'm_out_{layer}']
            proj = mlp_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    "example_index": row_i,
                    "prompt": prompt,
                    "subject": subject,
                    "gold_attribute": gold_attribute,
                    "attribute_tok": attribute_tok,
                    "attribute_tok_str": attribute_tok_str,
                    "attribute_tok_final_rank": attribute_tok_final_rank,
                    "layer": layer,
                    "proj_vec": "MLP",
                    "top_k_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                })

    res = pd.DataFrame.from_records(records)
    res["layer_1"] = res.layer.apply(lambda x: x+1)

    return res
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to data to be processed. Should contain the columns 'prompt' and 'subject'.",
    )
    argparser.add_argument(
        "--save_folder",
        required=True,
        type=str,
        help="Path to folder to save results to.",
    )
    argparser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Huggingface name of the model to get results for.",
    )
    argparser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Cache path for model weights.",
    )
    args = argparser.parse_args()
    print(args)
    
    if args.data_path.endswith('.json'):
        data = pd.read_json(args.data_path)
    elif args.data_path.endswith('.jsonl'):
        data = pd.read_json(args.data_path, lines=True)
    else:
        raise ValueError("Can only handle data files of type .json or .jsonl.")

    preds_data = get_intermediate_mhsa_mlp_preds(data, 
                                                 args.model_name,
                                                 args.cache_dir)
    data_name = Path(args.data_path).stem
    save_filename = f"intermediate_mhsa_mlp_preds_{data_name}_{args.model_name.split('/')[-1].replace('-', '_')}.jsonl"
    save_path = os.path.join(args.save_folder, save_filename)
    preds_data.to_json(save_path, lines=True, orient='records')
    
    print(f"Done! Intermediate predictions have been saved to '{save_path}'.")