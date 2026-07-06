#!/bin/bash

trap "echo -e '\n🛑 [Interrupt] Terminating overall loop...'; exit 1" SIGINT
# models=(
#     "meta-llama/Llama-2-7b-hf"
#     "meta-llama/Llama-2-13b-hf"
#     "meta-llama/Llama-3.1-8B"
# )
models=(
    "meta-llama/Llama-3.1-8B"
)

for model_name in "${models[@]}"; do
    echo "=================================================="
    echo "🚀 Starting Quantization & Eval for: ${model_name}"
    echo "=================================================="

    python main.py \
        --block_size 128 \
        --max_rotation_step 256 \
        --epochs 0 \
        --wbits 16 \
        --abits 16 \
        --model "$model_name" \
        --lwc \
        --lac 0.9 \
        --swc 0.8 \
        --eval_ppl \
        --multigpu \
        --permutation_times 0 \
        # --smooth \
        # --alpha 0.6 \
        # --only_r1 \
            # --task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa \
            
    echo "✅ Finished: ${model_name}"
    echo "=================================================="
done
            