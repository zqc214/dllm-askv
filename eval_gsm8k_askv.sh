#!/bin/bash
# GSM8K Evaluation with ASKV (Adaptive Spectral KV Compression)
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

echo "============================================"
echo "Fast-dLLM + ASKV Evaluation on GSM8K"
echo "============================================"

# 1. Dual Cache + ASKV
echo ""
echo "[1/2] Running Dual Cache + ASKV Compression..."
CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,use_spectral_compression=True,show_speed=True

# 2. Prefix Cache + ASKV
echo ""
echo "[2/2] Running Prefix Cache + ASKV Compression on GPU 1..."
CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,threshold=0.9,use_spectral_compression=True,show_speed=True

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"

