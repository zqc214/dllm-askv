#!/bin/bash
# HumanEval Evaluation with ASKV (Adaptive Spectral KV Compression)
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'

echo "============================================"
echo "Fast-dLLM + ASKV Evaluation on HumanEval"
echo "============================================"

# 1. Dual Cache + ASKV
echo ""
echo "[1/2] Running Dual Cache + ASKV Compression on GPU 1..."
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,use_spectral_compression=True,show_speed=True \
--output_path evals_results/dual_cache_askv/humaneval-ns0-${length} --log_samples

# 2. Prefix Cache + ASKV
echo ""
echo "[2/2] Running Prefix Cache + ASKV Compression on GPU 1..."
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,threshold=0.9,use_spectral_compression=True,show_speed=True \
--output_path evals_results/prefix_cache_askv/humaneval-ns0-${length} --log_samples

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo ""
echo "NOTICE: Use postprocess for HumanEval results:"
echo "  python postprocess_code.py evals_results/dual_cache_askv/humaneval-ns0-${length}/samples_*.jsonl"
echo "  python postprocess_code.py evals_results/prefix_cache_askv/humaneval-ns0-${length}/samples_*.jsonl"

