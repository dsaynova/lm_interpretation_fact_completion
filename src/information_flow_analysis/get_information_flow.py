# utilities
from src.information_flow_analysis.utils import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    trace_with_proj,
    trace_with_attn_block
)

import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

torch.set_grad_enabled(False)
tqdm.pandas()

def get_information_flow(data, model_name, cache_dir):
    # check what information is flowing to the last position states by intervening on attention edges
    print(f"Loading {model_name}...")
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        cache_dir=cache_dir
    )
    mt.model.eval()
    
    # Information flow analysis
    window = 9

    # Run attention knockouts
    results = []
    for row_i, row in tqdm(data.iterrows(), total=len(data), desc="Recording information flow"):
        prompt = row.prompt
        subject = row.subject

        inp = make_inputs(mt.tokenizer, [prompt])
        try:
            e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
            e_range = [x for x in range(e_range[0], e_range[1])]
        except ValueError as e:
            print(f"Encountered a problem processing entity range for sample {row_i}:")
            print(e)
            print("skipping...")
            continue

        # for the PRISM datasets, we want to intervene on what the answer_t should be
        if "rank_answers" in row.index:
            answer_t = mt.tokenizer.encode(row.prediction, add_special_tokens=False)
            assert len(answer_t) == 1
            answer_t = answer_t[0]
            answer_t_final_rank = row.rank_answers
        else:
            answer_t = None
            answer_t_final_rank = 0
            
        answer_t, base_score, _ = trace_with_proj(mt.model, inp, answer_t=answer_t)
        base_score = base_score.cpu().item()
        [answer] = decode_tokens(mt.tokenizer, [answer_t])

        ntoks = inp["input_ids"].shape[1]
        source_ = ntoks-1

        for block_ids, block_desc in [
            ([x for x in e_range], "subject"),
            ([x for x in range(ntoks-1) if x not in e_range], "non-subject"),
            ([source_], "last"),
        ]:
            for layer in range(mt.num_layers):
                layerlist = [
                    l for l in range(
                        max(0, layer - window // 2), min(mt.num_layers, layer - (-window // 2))
                    )
                ]
                block_config = {
                    l: [(source_, stok) for stok in block_ids]
                    for l in layerlist
                }
                r = trace_with_attn_block(
                    mt.model, inp, block_config, answer_t
                )
                new_score = r.cpu().item()
                results.append({
                    "example_index": row_i,
                    "prompt": prompt,
                    "answer": answer,
                    "answer_t": answer_t.cpu().item() if torch.is_tensor(answer_t) else answer_t,
                    "answer_t_final_rank": answer_t_final_rank,
                    "block_desc": block_desc,
                    "layer": layer,
                    "base_score": base_score,
                    "new_score": new_score,
                    "relative diff": (new_score - base_score) * 100.0 / base_score,
                    "is_subject_position_zero": e_range[0] == 0
                })

    res = pd.DataFrame.from_records(results)
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
        "--topk",
        default=50,
        type=int,
        help="Number of top intermediate preds to store per entry.",
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

    flow_data = get_information_flow(data, 
                                        args.model_name, 
                                        args.cache_dir)
    data_name = Path(args.data_path).stem
    save_filename = f"information_flow_{data_name}_{args.model_name.split('/')[-1].replace('-', '_')}.jsonl"
    save_path = os.path.join(args.save_folder, save_filename)
    flow_data.to_json(save_path, lines=True, orient='records')
    
    print(f"Done! Intermediate predictions have been saved to '{save_path}'.")