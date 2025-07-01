# coding=utf-8
# from here: https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/utils.py
# also added functions from the notebook under the same repository
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re
import functools

import torch
import transformers


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      cache_dir=None
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if model is None:
      assert model_name is not None
      model = transformers.AutoModelForCausalLM.from_pretrained(
          model_name, 
          low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype,
          cache_dir=cache_dir
          )
      set_requires_grad(False, model)
      model.eval().cuda()
    self.tokenizer = tokenizer
    self.model = model
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp, answer_t=None):
  # modified here: if answer_t is supplied, we wish to get the probabilities for the same token id
  # and return this as the "prediction"
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  if answer_t:
    preds = [answer_t]
    p = probs[:,answer_t]
  else:
    p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)

def _split_heads(tensor, num_heads, attn_head_size):
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(1, 0, 2)  # (head, seq_length, head_features)

def set_act_get_hooks(model, tok_index, attn=False, attn_out=False, mlp=False, mlp_coef=False):
    """
    Only works on GPT2
    """
    # Make sure that these are not set to True at the same time 
    #  so we don't put two different hooks on the same module.  
    assert not (attn is True and attn_out is True)
    
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "attn" in name:
                if "c_attn" in name:
                    # output.shape: batch_size, seq_len, 3 * hidden_dim
                    _, _, attn_value = output[0].split(model.config.n_embd, dim=1)
                    attn_value = _split_heads(attn_value,
                                              model.config.n_head, 
                                              model.config.n_embd // model.config.n_head)
                    model.activations_[name] = attn_value.detach()
                elif "attn_weights" in name:
                    assert len(output) == 3
                    attn_weights = output[2]  # (batch_size, num_heads, from_sequence_length, to_sequence_length)
                    # the last dimension is a distribution obtained from softmax
                    model.activations_[name] = attn_weights[0][:, tok_index, :].detach()
                else:
                    model.activations_[name] = output[0][:, tok_index].detach()
            elif "m_coef" in name:
                # num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                model.activations_[name] = input[0][:, tok_index].detach()
            elif "m_out" in name:
                model.activations_[name] = output[0][tok_index].detach()
        
        return hook

    hooks = []
    for i in range(model.config.n_layer):
        if attn is True:
            hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(get_activation(f"c_attn_value_{i}")))
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_weights_{i}")))
        if attn_out is True:
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_out_{i}")))
        if mlp_coef is True:
            hooks.append(model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i))))
        if mlp is True:
            hooks.append(model.transformer.h[i].mlp.register_forward_hook(get_activation("m_out_" + str(i))))
            
    return hooks
  
# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
        
def remove_wrapper(model, hooks):
    for i, hook in hooks:
        model.transformer.h[i].attn.forward = hook

# To block attention edges, we zero-out entries in the attention mask.
# To do this, we add a wrapper around the attention module, because 
# the mask is passed as an additional argument, which could not be fetched 
# with standard hooks before pytorch 2.0.  
def set_block_attn_hooks(model, from_to_index_per_layer, opposite=False):
    """
    Only works on GPT2
    """
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            hs = args[0]
            num_tokens = list(hs[0].size())[0]
            num_heads = model_.config.num_attention_heads
            
            if opposite_:
                attn_mask = torch.tril(torch.zeros((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 1
            else:
                attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 0
            attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
            
            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(hs.device)

            new_kwargs["attention_mask"] = attn_mask
            
            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn
    
    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.transformer.h[i].attn.forward
        model.transformer.h[i].attn.forward = wrap_attn_forward(model.transformer.h[i].attn.forward,
                                                                model, from_to_index_per_layer[i], opposite)
        hooks.append((i, hook))
    
    return hooks

def trace_with_attn_block(
    model,
    inp,
    from_to_index_per_layer,   # A list of (source index, target index) to block
    answers_t
):
    with torch.no_grad():
        # set hooks
        block_attn_hooks = set_block_attn_hooks(model, from_to_index_per_layer)
        
        # get prediction
        outputs_exp = model(**inp)
        
        # remove hooks
        remove_wrapper(model, block_attn_hooks)
    
    probs = torch.softmax(outputs_exp.logits[0, -1, :], dim=0)[answers_t]
    
    return probs

def set_proj_hooks(model):
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):
            num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
            if name == f"layer_residual_{final_layer}":
                hs = output
            else:
                hs = input[0]
            probs, preds = torch.max(
                torch.softmax(hs.matmul(E.T), dim=-1), 
                dim=-1
            )
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()
            model.projs_[f"{name}_probs"] = probs.cpu().numpy()
        return hook

    E = model.get_input_embeddings().weight.detach()
    final_layer = model.config.n_layer-1
    
    hooks = []
    for i in range(model.config.n_layer-1):
        hooks.append(model.transformer.h[i].register_forward_hook(
            get_projection(f"layer_residual_{i}", E)
        ))
    hooks.append(model.transformer.ln_f.register_forward_hook(
        get_projection(f"layer_residual_{final_layer}", E)
    ))

    return hooks

def trace_with_proj(model, inp, answer_t=None):
    with torch.no_grad():
        # set hooks
        hooks = set_proj_hooks(model)
        
        # get prediction/score
        if answer_t:
          answer_t, base_score = [d[0] for d in predict_from_input(model, inp, answer_t)]
        else:
          answer_t, base_score = [d[0] for d in predict_from_input(model, inp)]
        
        # remove hooks
        remove_hooks(hooks)
        
    projs = model.projs_
    
    return answer_t, base_score, projs