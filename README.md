# Evaluating the Impact of Verbal Multiword Expressions on Machine Translation

![ACL 2026](https://img.shields.io/badge/Accepted-ACL_2026_Main-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2508.17458-b31b1b.svg)](https://arxiv.org/abs/2508.17458)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository accompanies the paper **“Evaluating the Impact of Verbal Multiword Expressions on Machine Translation.”**

It provides:

- the **processed artifacts** used in the paper, released under [`preset/`](./preset)
- the **scripts required to rebuild datasets** from source materials
- the **pipelines to rerun the VMWE and WMT experiments**

The paper studies how **verbal multiword expressions (VMWEs)** affect machine translation, with a focus on:

- **VID** — verbal idioms
- **VPC** — verb-particle constructions
- **LVC** — light verb constructions

**Main finding:** VMWEs consistently reduce translation quality, and a substantial portion of that degradation is attributable to the VMWE itself rather than to overall sentence difficulty.

---

## Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Required Credentials](#required-credentials)
- [Experiment Scope](#experiment-scope)
- [Released Artifacts in `preset/`](#released-artifacts-in-preset)
- [Reproducing the Pipelines](#reproducing-the-pipelines)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

If you only want to inspect the released paper artifacts and do **not** need to run the code, start in [`preset/`](./preset).

Recommended entry points:

1. **Main VMWE dataset results**  
   [`preset/VMWE/MT_eval/`](./preset/VMWE/MT_eval)

2. **Paraphrase support analysis**  
   [`preset/VMWE/MT_para_eval/`](./preset/VMWE/MT_para_eval)

3. **WMT summary tables**  
   - [`preset/WMT/WMT_MT.csv`](./preset/WMT/WMT_MT.csv)
   - [`preset/WMT/WMT_Human.csv`](./preset/WMT/WMT_Human.csv)

4. **Per-example WMT subsets**  
   - [`preset/WMT/MT/`](./preset/WMT/MT)
   - [`preset/WMT/Human/`](./preset/WMT/Human)

For a file-by-file mapping to the paper, see [Released Artifacts in `preset/`](#released-artifacts-in-preset).

---

## Environment Setup

We recommend using a **`uv`-managed virtual environment**.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
uv pip install --no-build-isolation flash-attn==2.7.4.post1
```

If you prefer not to activate the environment manually, you can also use `uv run`.

### Dependency Notes

Some resources are downloaded automatically on first use, including:

- Hugging Face model weights
- COMET / MetricX resources
- `nltk`
- spaCy's `en_core_web_sm`

---

## Hardware Requirements

| Stage / Model | Recommended Minimum Hardware |
| :-- | :-- |
| **Main MT models** | 1 × NVIDIA RTX 6000 Ada (48 GB) per active translation job |
| **Paraphrase (`Llama-3.3-70B`)** | 3 × NVIDIA RTX 6000 Ada (48 GB each) |
| **MetricX evaluation** | 2 × NVIDIA RTX 6000 Ada |
| **XCOMET evaluation** | 1 × NVIDIA RTX 6000 Ada |
| **Dataset construction** | CPU-only is sufficient |

**Notes**

- Memory requirements may vary by checkpoint, `transformers` version, and parallelization strategy.
- If heavy stages are run sequentially, the same 3-GPU machine can usually be reused across paraphrase, MT, and QE by reallocating devices.
- If you cannot host `Llama-3.3-70B`, use `--paraphrase-model-id` to substitute another paraphrasing backend.

---

## Required Credentials

Some experiments require external credentials or gated access.

### Google Translate

Requires either:

- `GOOGLE_CLOUD_PROJECT`, or
- `--google-project-id`

You must also have valid Google Cloud credentials configured. See the [official Google Cloud authentication guide](https://docs.cloud.google.com/translate/docs/authentication#client-libraries).

### OpenAI-based Reclassification

Requires:

- `OPENAI_API_KEY`

### Gated Hugging Face Models

Some checkpoints require:

- accepted model licenses
- an authenticated Hugging Face session

---

## Experiment Scope

### Main VMWE Experiment

**Datasets**
- `LVC`
- `VPC`
- `VID`
- `Non_VMWE` (contrast set)

**Language directions**
- `en-cs`
- `en-de`
- `en-es`
- `en-ja`
- `en-ru`
- `en-tr`
- `en-zh`

**MT systems**
- `Google`
- `GemmaX2`
- `LLaMAX`
- `phi4`
- `Madlad`
- `M2M100`
- `opus`
- `seamless`

**QE models**
- `MetricX`
- `xCOMET`

### WMT Analysis

**Years**
- WMT 2017–2024

**Language pairs**
- `en-cs`
- `en-de`
- `en-ru`
- `en-zh`

**Comparison types**
- Human comparisons
- MT system comparisons

> `GPT-4.1` and `GPT-5.1` appear only as **support experiments** in `preset/VMWE/MT_eval/`. They are included in the released artifacts but are not exposed as runnable public backends in the reproduction scripts.

---

## Released Artifacts in `preset/`

The [`preset/`](./preset) directory contains the processed outputs used in the paper, including dataset derivatives and experiment outputs produced by the MT, evaluation, extraction, and summary pipelines.

### Mapping from Paper Sections to Released Files

| Paper section | Released artifact |
| :-- | :-- |
| **VMWE MT + QE results** (Sec. 4, 6) | `preset/VMWE/MT_eval/<model>/*.csv` |
| **Paraphrase analysis** (Sec. 7) | `preset/VMWE/MT_para_eval/<model>/*_{original,para,mixed}.csv` |
| **WMT candidate classification** (Sec. 5) | `preset/WMT/WMT_{LVC,VPC,VID}_Classified_2017_to_2024.csv` |
| **WMT MT summary results** (Sec. 5, 6) | `preset/WMT/WMT_MT.csv` |
| **WMT human summary** (Sec. 5, 6) | `preset/WMT/WMT_Human.csv` |
| **Per-example WMT MT** | `preset/WMT/MT/*.csv` |
| **Per-example WMT human** | `preset/WMT/Human/*.csv` |
| **Ranked MT systems** | `preset/WMT/WMT_system_rankings.csv` |

### File Naming Conventions

#### `preset/VMWE/MT_eval/`

Files follow:

```text
<DATASET>_<PAIR>.csv
```

Example:

```text
LVC_en-cs.csv
```

These files contain fields such as:

- `src`
- VMWE candidate columns
- `mt`
- `metricx_score`
- `xcomet_score`

#### `preset/VMWE/MT_para_eval/`

Files follow:

```text
<DATASET>_<PAIR>_<VIEW>.csv
```

Where `<VIEW>` is one of:

- `original` — original source → original MT
- `para` — paraphrased source → paraphrased MT
- `mixed` — original source → paraphrased MT

---

## Reproducing the Pipelines

If you want to rebuild the datasets and rerun the experiments from the provided scripts rather than using the released `preset/` artifacts, follow the steps below.

### 1. Build the VMWE Datasets

Downloads source resources and constructs the core CSV datasets:

- `LVC`
- `VPC`
- `VID`
- `Non_VMWE`
- `VID_dictionary`

```bash
python scripts/build_vmwe_datasets.py
```

### 2. Reproduce Main VMWE MT + QE Results

Runs the primary MT and evaluation pipeline.

```bash
python scripts/reproduce_vmwe_mt_eval.py \
  --stage all \
  --models GemmaX2 LLaMAX phi4 Madlad M2M100 opus seamless Google \
  --pairs en-cs en-de en-es en-ja en-ru en-tr en-zh \
  --datasets LVC VPC VID Non_VMWE \
  --metrics metricx xcomet
```

Useful resource-management flags include:

- `--translation-gpus`
- `--metricx-gpus`
- `--xcomet-gpus`
- `--parallel-jobs`
- `--google-project-id`

### 3. Reproduce the Paraphrase Support Experiment

```bash
python scripts/reproduce_vmwe_para_mt_eval.py \
  --stage all \
  --models GemmaX2 LLaMAX phi4 Madlad M2M100 opus seamless Google \
  --pairs en-cs en-de en-es en-ja en-ru en-tr en-zh \
  --datasets LVC VPC VID \
  --metrics metricx xcomet
```

### 4. Build WMT 2017–2024 Datasets

Downloads and restructures WMT data into:

```text
datasets/WMT/<year>/<Human|MT>/<pair>/<system>.csv
```

```bash
python scripts/build_wmt_datasets.py --years 2017 2018 2019 2020 2021 2022 2023 2024
```

### 5. Build the WMT VMWE Pipeline and Final Summaries

To reproduce the released paper-style outputs using the shipped preset classifications:

```bash
python scripts/build_wmt_vmwe_pipeline.py \
  --classification preset \
  --use-preset-final-summary
```

### 6. Optional: Full WMT Reclassification

To rerun WMT candidate classification from scratch using an API model:

```bash
python scripts/build_wmt_vmwe_pipeline.py \
  --classification api \
  --openai-model gpt-4o
```

This step requires `OPENAI_API_KEY`.

> Outputs from API-based reclassification may not be byte-identical to the released preset files.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{liu2025evaluatingimpactverbalmultiword,
      title={Evaluating the Impact of Verbal Multiword Expressions on Machine Translation}, 
      author={Linfeng Liu and Saptarshi Ghosh and Tianyu Jiang},
      year={2025},
      eprint={2508.17458},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.17458}, 
}
```

---

## License

This project is released under the **MIT License**.
