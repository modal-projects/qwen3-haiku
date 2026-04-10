# Qwen3-4B Haiku -- slime GRPO training on Modal

Finetunes [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) to write haiku poems using [slime](https://github.com/THUDM/slime) GRPO on [Modal](https://modal.com).

## Prerequisites

- Modal CLI installed and authenticated (`pip install modal && modal setup`)
- Set your Modal environment: `export MODAL_ENVIRONMENT=<your-env>`

## Quick start

Three entrypoints cover the full workflow:

```bash
# 1. Generate configs, download model, prepare dataset, deploy LLM judges
modal run modal_train.py::prepare

# 2. Kick off all training runs (7 judge configurations in parallel)
modal run modal_train.py::train

# 3. Deploy model servers and run evals
modal run modal_train.py::evaluate
```

## Running a single experiment

All experiments are config-driven. List available configs:

```bash
modal run modal_train.py::list_configs
```

Run steps individually via `EXPERIMENT_CONFIG`:

```bash
# Download model (one-time)
EXPERIMENT_CONFIG=qwen3_4b_haiku modal run modal_train.py::_download_model

# Prepare dataset (one-time)
EXPERIMENT_CONFIG=qwen3_4b_haiku modal run modal_train.py::_prepare_dataset

# Train (detached)
EXPERIMENT_CONFIG=qwen3_4b_haiku modal run -d modal_train.py::_train_single
```

## Experiment configs

Each experiment is a Python module in `configs/` exposing `modal` (infrastructure) and `slime` (training args) instances.

| Config | Reward model |
|---|---|
| `qwen3_4b_haiku` | Structure-only (no LLM judge) |
| `qwen3_4b_haiku_standard_4b` | Standard judge, Qwen3-4B (self-judge) |
| `qwen3_4b_haiku_standard_30b` | Standard judge, Qwen3-30B |
| `qwen3_4b_haiku_standard_235b` | Standard judge, Qwen3-235B |
| `qwen3_4b_haiku_cl_4b` | Curriculum learning judge, Qwen3-4B (self-judge) |
| `qwen3_4b_haiku_cl_30b` | Curriculum learning judge, Qwen3-30B |
| `qwen3_4b_haiku_cl_235b` | Curriculum learning judge, Qwen3-235B |

Judge variant configs are generated from the base config:

```bash
python configs/generate_judge_variants.py
```

## Adding a new experiment

Create `configs/<your_experiment>.py`:

```python
from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH

modal = ModalConfig(gpu="H200")

class _Slime(SlimeConfig):
    hf_checkpoint = "Qwen/Qwen3-4B"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True
    # ... all other slime args as snake_case attributes

    def prepare_data(self) -> None:
        # download and preprocess your dataset
        ...

slime = _Slime()
```

Every attribute on `_Slime` (except `environment`, `async_mode`, `slime_model_script`) is forwarded to slime as a CLI argument: `field_name` -> `--field-name`.

## Project structure

```
modal_train.py              # Main launcher (prepare / train / evaluate)
config.py                   # Judge type enums (shared with llm_judges)
configs/
  base.py                   # ModalConfig + SlimeConfig base classes
  qwen3_4b_haiku.py         # Base haiku experiment
  qwen3_4b_haiku_*.py       # Judge variant configs (generated)
  generate_judge_variants.py
llm_judges/
  deploy.py                 # LLM judge endpoints (modal deploy)
  base.py                   # HaikuJudge scoring logic
  nlp.py                    # Syllable counting / structure scoring
eval/
  serve_haiku_model.py      # vLLM model serving (modal deploy)
  run_eval.py               # Evaluation runner
  haiku_app.py              # Interactive playground UI
  shared.py                 # Model registry and helpers
tools/
  convert_torch_dist_to_hf.py  # Megatron -> HuggingFace checkpoint conversion
```
