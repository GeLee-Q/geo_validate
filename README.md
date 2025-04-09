# geo_validate

本文档概述了使用 Qwen2.5-VL-7B-Instruct（sglang） 来验证 `geo3k` 数据集的步骤。


#### **启动 sglang serve**
```bash
bash qwen2_5_vl_sglang_server.sh
```

默认端口是 30000
在另外一个终端启动 client 验证 sglang

#### **启动 client 验证 sglang**
```bash
python geo3k_validate_client.py --port 30000 --save-mode full --tag sglang
```

#### **启动 vllm serve**
```bash
bash qwen_2_5_vl_vllm_server.sh
```
默认端口是 8080
在另外一个终端启动 client 验证 vllm

#### **启动 client 验证 vllm**
```bash
python geo3k_validate_client.py --port 8080 --save-mode full --tag vllm
```

最后结果会保存在 `evaluation_results_vllm.csv` 和 `evaluation_results_sglang.csv` 文件中
log 文件会保存在类似于 `geo3k_validation_engine_xxx.log` 的文件中，其中最下面有验证的统计信息

