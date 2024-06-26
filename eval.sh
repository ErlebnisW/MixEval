export CUDA_VISIBLE_DEVICES=0

python -m mix_eval.evaluate \
    --model_name gemma_2b\
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 4 \
    --max_gpu_memory 40GiB \
    --output_dir mix_eval/data/model_responses/ \
    --api_parallel_num 10
    