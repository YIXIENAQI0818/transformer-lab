#!/usr/bin/env bash
# 步骤 5：把 HF 模型导入 Ollama（用 --experimental 直接读 safetensors）
#
# 原计划是用 llama.cpp 的 convert_hf_to_gguf.py 转成 GGUF 再给 Ollama，但两件事改变了路径：
#   1. clone llama.cpp 因网络不稳定（关代理直连 GitHub 反复断流）失败；
#   2. 发现 Ollama 0.33+ 的 `--experimental` 标志能直接读 safetensors（内部自动转换），
#      省去手动 GGUF 那一步。
#
# 所以实际路径是：out/hf（safetensors + config + tokenizer）--ollama create--> Ollama 模型。
#
# 已知障碍（如实记录）：
#   Ollama 0.33.2 的 safetensors import 在 Linux 上依赖 MLX（Apple 的框架），运行/量化时
#   报 "MLX not available"。这是 Ollama 的版本环境限制，不是迁移本身的问题。
#   绕过方向：手动写 GGUF（用 pip 的 gguf 包，绕开 Ollama 的 MLX 量化器），
#   标准 GGUF 走 llama.cpp runner（本机 qwen2.5:3b 就是 GGUF，能正常跑）。
#
# 用法（在 hf-export/ 目录下）：
#   bash src/05_import_ollama.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

echo "=== 1. 导入 safetensors（Ollama 自动转 GGUF）==="
ollama create --experimental my-gpt -f Modelfile

echo "=== 2. 确认模型已注册 ==="
ollama list | grep my-gpt || true

echo "=== 3. 运行生成（本机需 Ollama 能识别 my-gpt；若 server 未刷新见 README 说明）==="
ollama run my-gpt "ROMEO:"
