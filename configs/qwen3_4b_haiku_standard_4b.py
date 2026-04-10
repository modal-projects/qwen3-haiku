"""Qwen3-4B Haiku -- Standard judge, Qwen3-4B (self-judge)."""

from configs import qwen3_4b_haiku as _base

modal = _base.modal


class _Slime(_base._Slime):
    rm_type = "remote_rm"
    rm_url = "https://modal-labs-joy-dev--llm-judge-standard4b.us-east.modal.direct/score"
    custom_rm_path = None

    wandb_group = "qwen3-4b-haiku-standard-4b"


slime = _Slime()
