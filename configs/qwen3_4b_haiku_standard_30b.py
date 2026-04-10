"""Qwen3-4B Haiku -- Standard judge, Qwen3-30B-A3B judge model."""

from configs import qwen3_4b_haiku as _base

modal = _base.modal


class _Slime(_base._Slime):
    rm_type = "remote_rm"
    rm_url = "https://modal-labs-joy-dev--llm-judge-standard30b.us-east.modal.direct/score"
    custom_rm_path = None

    wandb_group = "qwen3-4b-haiku-standard-30b"


slime = _Slime()
