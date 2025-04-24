python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 30201 \
    --chunked-prefill-size -1 \
    --chat-template qwen2-vl \
    "$@"
