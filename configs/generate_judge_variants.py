#!/usr/bin/env python3
"""Generate judge variant config files from the base haiku config.

Usage:
    python configs/generate_judge_variants.py
"""

from pathlib import Path

CONFIGS_DIR = Path(__file__).parent

# Base config module name (must already exist)
BASE_CONFIG = "qwen3_4b_haiku"

# Judge endpoint URL pattern -- {cls_name} is lowercased class name (e.g. "standard30b")
JUDGE_URL_TEMPLATE = (
    "https://modal-labs-joy-dev--llm-judge-{cls_name}.us-east.modal.direct/score"
)

# (file_suffix, judge_cls_name, description)
VARIANTS = [
    ("standard_4b",   "standard4b",   "Standard judge, Qwen3-4B (self-judge)"),
    ("standard_30b",  "standard30b",  "Standard judge, Qwen3-30B-A3B judge model"),
    ("standard_235b", "standard235b", "Standard judge, Qwen3-235B judge model"),
    ("cl_4b",         "cl4b",         "Curriculum learning judge, Qwen3-4B (self-judge)"),
    ("cl_30b",        "cl30b",        "Curriculum learning judge, Qwen3-30B-A3B judge model"),
    ("cl_235b",       "cl235b",       "Curriculum learning judge, Qwen3-235B judge model"),
]

TEMPLATE = '''\
"""Qwen3-4B Haiku -- {description}."""

from configs import {base} as _base

modal = _base.modal


class _Slime(_base._Slime):
    rm_type = "remote_rm"
    rm_url = "{rm_url}"
    custom_rm_path = None

    wandb_group = "{wandb_group}"


slime = _Slime()
'''


def main():
    for suffix, cls_name, description in VARIANTS:
        filename = f"{BASE_CONFIG}_{suffix}.py"
        path = CONFIGS_DIR / filename
        content = TEMPLATE.format(
            description=description,
            base=BASE_CONFIG,
            rm_url=JUDGE_URL_TEMPLATE.format(cls_name=cls_name),
            wandb_group=f"{BASE_CONFIG.replace('_', '-')}-{suffix.replace('_', '-')}",
        )
        path.write_text(content)
        print(f"  wrote {filename}")

    print(f"\nGenerated {len(VARIANTS)} variant configs.")


if __name__ == "__main__":
    main()
