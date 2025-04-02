from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process

vision_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path /workspace/Qwen2.5-VL-7B-Instruct \
    --chat-template=qwen2-vl
"""
)

wait_for_server(f"http://localhost:{port}")