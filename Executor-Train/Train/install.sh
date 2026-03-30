#!/bin/bash

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
pip install "sglang[all]==0.4.10.post2" --no-cache-dir && pip install "flashinfer-python==0.2.9rc2" "torch_memory_saver==0.0.8" --no-cache-dir 
pip install --no-cache-dir "vllm==0.10.1.1" "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" "tensordict==0.9.1" "torchdata==0.11.0"

echo "2. install basic packages"
pip install "transformers==4.55.4" accelerate datasets peft hf-transfer \
    numpy "pyarrow==20.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff tensorboard

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"



echo "Successfully installed all packages"