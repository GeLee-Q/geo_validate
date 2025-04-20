#!/bin/bash
source ~/.python/sglang/bin/activate

export CUDA_VISIBLE_DEVICES=7

python3 -m sglang.launch_server \
    --model-path /sgl-workspace/models/Qwen2.5-VL-7B-Instruct \
    --chat-template qwen2-vl \
    --disable-radix-cache \
    --port 40912 \
    --context-length 4096
        