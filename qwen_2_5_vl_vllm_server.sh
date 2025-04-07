

export CUDA_VISIBLE_DEVICES=7

python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --max-num-seqs 32 \
    --trust-remote-code \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching