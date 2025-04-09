# geo_validate

本文档概述了使用 Qwen2.5-VL-7B-Instruct 来对齐 SGLang 和 vllm 在 geo3k 数据集上的精度遇到的问题。
为了得到 SGLang 的最高精度，我们 disable 了 chunk prefill 和 prefix cache，同时对 SGLang 的 max running requests 也设置为了 1。这导致 SGLang 的运行速度相当慢。即便如此，得到的结果也不尽理想。

## 在 atlas 上安装环境

启动 docker：

```bash
docker run -it --name h100_VLM_calibration --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

一把子安装所有环境：

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
sudo apt install tmux -y
python3 -m venv ~/.python/sglang
source ~/.python/sglang/bin/activate
python3 -m pip install uv
python3 -m uv pip install gpustat

# Install latest SGlang from main branch
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
python3 -m uv pip install vllm==0.8.3


python3 -m uv pip install mathruler
python3 -m uv pip install loguru
python3 -m uv pip install pandas
python3 -m uv pip install tqdm
python3 -m uv pip install pylatexenc

git clone -b txy/add_engine_sgl https://github.com/GeLee-Q/geo_validate.git
cd geo_validate
```



## 测试 SGLang

开一个 tmux 会话，启动 SGLang：

```bash
export CUDA_VISIBLE_DEVICES=0
bash qwen2_5_vl_sglang_server.sh
```

默认端口是 30201，在另外一个终端启动 client 验证 SGLang：

```bash
python geo3k_validate_client.py --port 30201 --save-mode full --tag sglang
```

## 测试 vllm

开一个 tmux 会话，启动 vllm：

```bash
export CUDA_VISIBLE_DEVICES=1
bash qwen_2_5_vl_vllm_server.sh
```

默认端口是 8071，在另外一个终端启动 client 验证 vllm：

```bash
python geo3k_validate_client.py --port 8071 --save-mode full --tag vllm
```

## 测试结果

SGLang 由于设置了最高精度，所以运行速度相当慢，即便如此，效果也不佳：


```bash
2025-04-09 16:44:33 | INFO | Total processing time: 1969.41 seconds
2025-04-09 16:44:33 | INFO | Average time per row: 3.28 seconds
2025-04-09 16:44:33 | INFO | The mean score is: 0.33277870216306155
2025-04-09 16:44:33 | INFO | Results saved to evaluation_results_sglang.csv in full mode
2025-04-09 16:44:33 | INFO | Processing completed successfully
```

vllm 效果确实不错：

```bash
2025-04-09 16:21:23 | INFO | Total processing time: 140.89 seconds
2025-04-09 16:21:23 | INFO | Average time per row: 0.23 seconds
2025-04-09 16:21:23 | INFO | The mean score is: 0.3828618968386024
2025-04-09 16:21:23 | INFO | Results saved to evaluation_results_vllm.csv in full mode
2025-04-09 16:21:23 | INFO | Processing completed successfully
```

最后结果会保存在 `evaluation_results_vllm.csv` 和 `evaluation_results_sglang.csv` 文件中
log 文件会保存在类似于 `geo3k_validation_engine_xxx.log` 的文件中，其中最下面有验证的统计信息。
