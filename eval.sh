export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT="https://hf-mirror.com"

name=qwen2-2b_alpaca_sft
model_name=qwen_2_7b_instruct
model_path=/data1/WM_workspace/MDSPO/output/qwen2-1.5b_alpaca_sft

python -m mix_eval.evaluate \
    --model_name ${model_name}\
    --model_path ${model_path} \
    --benchmark mixeval_hard \
    --multichoice_judge custom \
    --freeform_judge custom \
    --version 2024-06-01 \
    --batch_size 2 \
    --max_gpu_memory 40GiB \
    --output_dir output/${name} \
    --api_parallel_num 10
    