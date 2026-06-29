import torch
import torch.nn as nn
import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    AutoTokenizer,
    AutoConfig
)
import transformers

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")

    args = parser.parse_args()
    model=LlamaForCausalLM.from_pretrained(args.model,device_map='auto')
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.model
    )

    layers = model.model.layers


    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # self.is_llama = False
            # self.is_qwen = False

            # Qwen3 compatibility
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]

            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            cache["cache_position"] = kwargs.get("cache_position", None)

            raise ValueError
