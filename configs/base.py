"""Base configuration classes and volume mount paths for slime training.

Two separate concerns:
  ModalConfig  -- Modal infrastructure (GPU model, image patches, dev overlay)
  SlimeConfig  -- slime training arguments
"""

import math
from pathlib import Path
from typing import Any, Literal

# -- Volume mount paths
HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# -- Types
GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

# Fields on SlimeConfig that are NOT slime CLI args.
_slime_SKIP = {"environment", "async_mode", "slime_model_script"}

# SlimeConfig fields that slime reads as YAML files at runtime.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class ModalConfig:
    """Modal infrastructure configuration -- GPU provisioning and image setup only."""

    gpu: GPUType = "H200"
    local_slime: str | None = None
    patch_files: list[str] = []
    image_run_commands: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class SlimeConfig:
    """Base slime training configuration.

    Subclass and set class attributes to configure an experiment.
    All attributes (except those in _slime_SKIP) are forwarded to slime as CLI args.

    Fields in _slime_SKIP are launcher instructions, not slime CLI args:
      environment        -- injected into the Ray job runtime env
      async_mode         -- selects train_async.py vs train.py
      slime_model_script -- path relative to /root/slime to a shell script that
                            defines MODEL_ARGS for model architecture
    """

    environment: dict = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    async_mode: bool = False
    slime_model_script: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _fields(self) -> dict[str, Any]:
        """Merged field dict from the class hierarchy; instance attrs win."""
        fields: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            fields.update(
                {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)}
            )
        fields.update(vars(self))
        return {k: v for k, v in fields.items() if k not in _slime_SKIP}

    def cli_args(self) -> list[str]:
        """slime CLI arguments derived from this config.

        Conversion rules:
          field_name -> --field-name  (underscore to hyphen)
          True       -> --flag        (no value)
          False/None -> omitted
          list       -> --flag v1 v2 ...
          other      -> --flag value
        """
        out: list[str] = []
        for key, val in self._fields().items():
            if val is None or val is False:
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    def prepare_data(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no prepare_data()")

    def total_nodes(self) -> int:
        """Total Modal cluster nodes required by this config."""
        f = self._fields()
        gpus_per_node = f.get("actor_num_gpus_per_node", 8)
        actor_nodes = f.get("actor_num_nodes", 1)
        colocate = f.get("colocate", False)
        use_critic = f.get("use_critic", False)
        critic_nodes = f.get("critic_num_nodes") or actor_nodes
        critic_gpus = f.get("critic_num_gpus_per_node") or gpus_per_node
        rollout_gpus = f.get("rollout_num_gpus")

        training_gpus = actor_nodes * gpus_per_node
        if use_critic:
            training_gpus += critic_nodes * critic_gpus

        if colocate:
            total_gpus = training_gpus
        else:
            rollout_gpus = rollout_gpus or (actor_nodes * gpus_per_node)
            total_gpus = training_gpus + rollout_gpus

        if total_gpus % gpus_per_node != 0:
            raise ValueError(
                f"total_gpus={total_gpus} is not a multiple of gpus_per_node={gpus_per_node}."
            )

        return math.ceil(total_gpus / gpus_per_node)
