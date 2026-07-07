
trap "echo -e '\n🛑 [Interrupt] Terminating overall loop...'; exit 1" SIGINT
# models=(
#     "meta-llama/Llama-2-7b-hf"
#     "meta-llama/Llama-2-13b-hf"
# )
models=(
    "meta-llama/Llama-3.1-8B"
)

LOG_DIR="./logs_quantization"
mkdir -p "${LOG_DIR}"

for model_name in "${models[@]}"; do

    pure_model_name=$(basename "${model_name}")

    echo "=================================================="
    echo "🚀 Starting Quantization & Eval for: ${pure_model_name}"
    echo "=================================================="

    # 💡 max_rotation_step 리스트를 순회하는 내부 루프 추가
    # for step in 32 64 128 256 512 1024; do
    for step in 16; do
        CURRENT_LOG="${LOG_DIR}/${pure_model_name}_step${step}.log"

        echo "--------------------------------------------------"
        echo "⚙️ Running with --max_rotation_step ${step}"
        echo "--------------------------------------------------"

        python main.py \
            --block_size -1 \
            --max_rotation_step "${step}" \
            --epochs 0 \
            --wbits 4 \
            --abits 4 \
            --model "$model_name" \
            --lwc \
            --lac 0.9 \
            --swc 0.8 \
            --eval_ppl \
            --multigpu \
            --permutation_times 0 \
            --only_r1 2>&1 | tee "${CURRENT_LOG}"

    done
done