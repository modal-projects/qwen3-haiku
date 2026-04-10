"""Qwen3-4B Haiku -- Curriculum learning judge, Qwen3-4B (self-judge)."""

from configs import qwen3_4b_haiku as _base

modal = _base.modal


class _Slime(_base._Slime):
    rm_type = "remote_rm"
    rm_url = "https://modal-labs-joy-dev--llm-judge-cl4b.us-east.modal.direct/score"
    custom_rm_path = None

    wandb_group = "qwen3-4b-haiku-cl-4b"


slime = _Slime()
