# geo_validate

本文档概述了使用 Qwen2.5-VL-7B-Instruct（sglang） 来验证 `geo3k` 数据集的步骤。

## 服务器配置

### 1.  **启动服务器:**

#### **启动 sglang serve**
```bash
python qwen2_5_vl_sglang_server.py
```

#### **启动 vllm serve**
```bash
bash qwen_2_5_vl_vllm_server.sh
```

### 2.  **参数配置:**

*   `--model-path /workspace/Qwen2.5-VL-7B-Instruct`: 指定本地存储的 Qwen2.5-VL-7B-Instruct 模型的路径。请根据您的模型实际存放位置调整此路径。

## 客户端验证

###  **启动客户端:**

有两种验证方式可供选择：

1. **使用 HTTP API 客户端:**
```bash
python geo3k_validate_client_2.py --port 8080
```
- 支持自定义端口，默认端口是 8080
- client_2 并行发送请求，验证速度相比较 geo_validate/geo3k_validate_client.py 大幅度加快

2. **使用 SGLang Engine 客户端:**
```bash
python geo3k_validate_engine_sgl.py
```
- 直接使用 SGLang Engine 进行验证，支持并行处理和更高效的推理

###  **参数配置:**

*   `PARQUET_FILE_PATH = "/workspace/geo3k/test.parquet"`:  设置此变量为您的 `geo3k` 数据集的 Parquet 文件的正确路径。
*   `LLM_ENDPOINT = "http://localhost:39125/v1/chat/completions"`: 配置此变量以匹配您启动的服务器实例的端口号。
*   `LLM_MODEL = "/workspace/Qwen2.5-VL-7B-Instruct"`: 确保此变量与启动服务器时使用的 `--model-path` 相匹配。
* `DEMO_ROW_LIMIT` 用来方便调试的，测试的数据是 DEMO_ROW_LIMIT + 1 条，后续删除


