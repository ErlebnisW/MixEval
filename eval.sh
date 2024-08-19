export HF_ENDPOINT="https://hf-mirror.com"
# export MODEL_PARSER_API=sk-Mo26BRz5h54k5k5RnqEzdnAIinN5bjDqSB3zSskL3ntsGvw1
export MODEL_PARSER_API=sk-0cy42rxb9fcXCswGA1E3F598C3A4435a8cA7A79b02E067Fb
# export MODEL_PARSER_API=sk-65edd2f3889f4ed1bc6f1a3269525819
# OPENAI_API_BASE=https://api.chatanywhere.tech/v1

base_url=https://xiaoai.plus/v1
model_path=/data/wmz_workspace/RT-SPPO/checkpoints/Llama-3-8B-Instruct-SPPO-Iter3
output_dir=mix_eval/data/model_responses/rt-sppo-iteration3

python -m mix_eval.evaluate \
    --model_name local_chat \
    --model_path ${model_path} \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --max_gpu_memory 30GiB \
    --output_dir ${output_dir}\
    --api_parallel_num 20 \
    --api_base_url ${base_url} \
    # --multichoice_judge gpt-3.5-turbo-ca \
    # --freeform_judge gpt-3.5-turbo-ca \
    # --api_base_url https://api.deepseek.com \
    
