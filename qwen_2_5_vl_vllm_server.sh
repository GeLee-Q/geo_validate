python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8071 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --trust-remote-code \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching
