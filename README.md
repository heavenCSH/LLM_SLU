# PICD-Instruct: A Generative Instruction Learning Framework for Few-Shot Multi-Intent Spoken Language Understanding

This repo provides the source code & data of our paper. For convenience, the training data for PICD-Instruct can be downloaded from [Baidu NetDisk](https://pan.baidu.com/s/1FihuCpWI9QZ9If4zQVhjSQ?pwd=qzfb).

## 1. Training PICD-Instruct

```bash
cd LLM_SLU/LLaMAFactory
python train.py
```

## 2. Inference
```bash
cd LLM_SLU
python -m eval_tool.slu_eval
```

## 3. Evaluation
```bash
cd LLM_SLU/eval_tool
python slu_calculate.py
```