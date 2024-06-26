export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m vllm.entrypoints.openai.api_server \
        --served-model-name='qwen2-72b' \
        --model="/data1/WM_workspace/checkpoints/qwen2-72b/" \
        --dtype=auto \
        --tensor-parallel-size=8 \
        --gpu-memory-utilization 0.6 \