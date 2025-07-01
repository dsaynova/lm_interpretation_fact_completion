# takes inspiration from ROME code
# note that GPT-2 XL does not support batch sizes larger than 1 since it does not function with padding.

import argparse
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
            
    aa(
        "--model_name",
        default="gpt2-xl",
        choices=[
            "gpt2-xl",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "EleutherAI/pythia-6.9b",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-32B"
        ],
    )
    aa("--srcfile", required=True)
    aa("--outfile", required=True)
    aa("--top_k", required=True, type=int)
    aa("--cache_folder", default=None, help="The cache directory for Huggingface.")
    args = parser.parse_args()
    print("Arguments parsed.")

    print("Loading LM...")
    mt = ModelAndTokenizer(args.model_name, cache_folder=args.cache_folder)
    
    print(f"Saving model predictions to {args.outfile}...")
    # get model preds                
    with open(args.outfile, "w") as f_out:
        with open(args.srcfile, "r") as f_in:
            # currently run with a batch size of 1 (GPT-2 XL does not support padding)
            for line in tqdm(f_in.readlines()):
                data = json.loads(line)
                with torch.no_grad():
                    inputs = mt.tokenizer(data["prompt"], return_tensors="pt").to("cuda")
                    answers_t, base_scores = predict_from_input(mt.model, inputs, k=args.top_k)
                    answers = [mt.tokenizer.decode(answer_t, skip_special_tokens=True) for answer_t in answers_t[0]]
                    data["answers"] = answers
                    data["p_answers"] = base_scores[0].tolist()
                    
                    f_out.write(json.dumps(data))
                    f_out.write("\n")
            
def predict_from_input(model, inp, k):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.topk(probs, k=k, dim=1)
    return preds, p

# copied from the ROME repo and slightly edited to remove CT stuff
class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        cache_folder=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True, cache_dir=cache_folder)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, 
                torch_dtype=torch_dtype, use_auth_token = True, cache_dir=cache_folder
            )
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        
if __name__ == "__main__":
    main()