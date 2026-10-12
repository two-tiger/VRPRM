# VRPRM: Process Reward Modeling via Visual Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2508.03556-b31b1b)](https://arxiv.org/abs/2508.03556)
[![GitHub](https://img.shields.io/badge/GitHub-VRPRM-black)](https://github.com/two-tiger/VRPRM)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

**VRPRM** is a family of visual reasoning process reward models (PRMs) for
multimodal mathematics and reasoning. Given an image, a problem, and a candidate
solution, VRPRM first constructs an explicit reasoning trace and then emits a
binary judgment for each intermediate step. The released models are built from
**Qwen3-VL-8B-Thinking** with:

- **Reasoning PRM formulation:** step-level rewards are conditioned on a generated
  `<think>...</think>` trace rather than produced by a score-only classifier.
- **Rollout-filtered distillation:** 1,378 multi-turn conversations and 5,592
  confident step labels are distilled from Kimi K2.6 and checked against
  conservative VisualPRM400K Monte-Carlo labels.
- **Weighted process supervision:** global-thinking tokens use loss weight 0.5,
  step-judgment tokens use loss weight 2.0, and uncertain steps are masked.
- **Process-supervised RL:** GSPO-token/GRPO LoRA RL uses 40K confident
  non-reasoning examples and a 0.97/0.02/0.01 step/format/think reward.
- **Reproducible evaluation:** VisualProcessBench uses the official no-reference
  prompt protocol and reports pooled `overall_step_macro_f1`.

## News

- **2026-10-12** — Public code, `VRPRM-sft`, `VRPRM-rl`, and the SFT dataset release.

## Results

VisualProcessBench step-level macro-F1 under the official no-reference protocol:

| Model | DynaMath | MMMU | MathVerse | MathVision | WeMath | Overall |
|---|---:|---:|---:|---:|---:|---:|
| VisualPRM-8B re-evaluated | – | – | – | – | – | 62.18 |
| Base + non-CoT RL control | 60.07 | 56.77 | 63.80 | 57.79 | 60.63 | 61.22 |
| **VRPRM-sft** | 66.99 | 60.52 | 66.86 | 65.61 | 64.31 | **66.15** |
| **VRPRM-rl** | 64.62 | 60.57 | 65.51 | 66.01 | 64.11 | 65.41 |

The single official Overall column is `overall_step_macro_f1`. The unweighted
mean over five subsets is a diagnostic only and is not used as a second paper
result. Full conventions are defined in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Released artifacts

| Artifact | Description |
|---|---|
| `VRPRM-sft` | Merged SFT checkpoint after 3 epochs / 513 steps. |
| `VRPRM-rl` | Merged process-supervised RL checkpoint used for the public release. |
| `VRPRM-sft-data` | Final multi-turn SFT dataset plus accepted/rejected teacher rollouts. |

The public RL artifact is the end-of-run checkpoint. See
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the exact checkpoint and validation
convention.

## Repository layout

```text
data_pipeline/       Rollout filtering and multi-turn SFT dataset construction
sft/                 SFT, LoRA merge, and SFT ablation launchers
easyr1/              EasyR1 fork used for GSPO/GRPO process RL
benchmarks/          VisualProcessBench and best-of-N evaluation
analysis/            Paper-table and training-curve generators
docs/                Protocol and data-generation details
ms-swift/            Pinned ms-swift runtime used by the SFT pipeline
```

Large checkpoints, generated datasets, caches, logs, and benchmark outputs are
intentionally excluded from Git.

## Installation

Python 3.10/3.11, PyTorch 2.8, CUDA 12.8, and vLLM >= 0.11 are recommended.
The SFT and RL trainers have different dependency constraints, so using two
environments is the safest setup.

```bash
git clone https://github.com/two-tiger/VRPRM.git
cd VRPRM

# Data generation and evaluation utilities
python -m pip install -r requirements.txt

# SFT
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r sft/requirements-sft.txt
python -m pip install flash-attn --no-build-isolation

# RL / EasyR1
python -m pip install -e easyr1
```

## Data preparation

Download [VisualPRM400K-v1.1-Raw](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1-Raw)
and place it under:

```text
data/VisualPRM400K-v1.1-Raw/
```

To reuse the released SFT dataset, download `VRPRM-sft-data` and place the final
JSON at:

```text
rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35.json
```

To regenerate it with an OpenAI-compatible Kimi endpoint:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=EMPTY \
bash data_pipeline/run_all.sh
```

## Training

### SFT cold start

```bash
THINKING_MODEL_PATH=Qwen/Qwen3-VL-8B-Thinking \
THINKING_MULTITURN_DATASET_PATH=rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35.json \
USE_TORCHRUN=true NPROC_PER_NODE=8 \
bash sft/train_vrprm_sft.sh

bash sft/merge_lora.sh \
  sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft/checkpoint-513
```

### Process RL

Set `MODEL_PATH` to the merged SFT checkpoint or to `VRPRM-sft`:

```bash
MODEL_PATH=/path/to/VRPRM-sft \
DATA_ROOT=data/VisualPRM400K-v1.1-Raw \
N_GPUS_PER_NODE=8 \
bash easyr1/examples/vrprm/train_vrprm_rl.sh
```

`PREPARE_ONLY=true` builds the filtered 40K/2K RL split without launching
training.

## Evaluation

Serve a merged model with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 \
bash benchmarks/visualprocessbench/serve_vllm.sh /path/to/VRPRM-sft
```

Run the official no-reference VisualProcessBench protocol:

```bash
cd benchmarks/visualprocessbench
VPB_MODE=global_think VPB_CONCURRENCY=8 bash run_eval_vpb.sh
```

For best-of-N selection, see [`benchmarks/vlmeval_bon/README.md`](benchmarks/vlmeval_bon/README.md).

## Protocol and reproducibility

All label thresholds, effective hyperparameters, metric definitions, and
checkpoint conventions are centralized in:

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md)
- [`docs/DATA_GENERATION.md`](docs/DATA_GENERATION.md)

The paper and model cards must follow these definitions.

## Citation

If you use VRPRM, please cite:

```bibtex
@article{chen2026vrprm,
  title   = {VRPRM: Process Reward Modeling via Visual Reasoning},
  author  = {Chen, Xinquan and Yue, Chongying and Liu, Bangwei and
             Wang, Xuhong and Wang, Yingchun and Lu, Chaochao},
  journal = {arXiv preprint arXiv:2508.03556},
  year    = {2026}
}
```

## Acknowledgements

VRPRM builds on [EasyR1](https://github.com/hiyouga/EasyR1),
[ms-swift](https://github.com/modelscope/ms-swift),
[VisualPRM](https://github.com/OpenGVLab/VisualPRM), and
[VisualPRM400K](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1-Raw).
We thank their authors for open research and reproducible tooling.

## License

Code is released under Apache License 2.0. Vendored projects retain their
original licenses. Model and dataset use is additionally subject to Qwen and
VisualPRM400K upstream terms.
