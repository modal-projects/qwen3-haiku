"""
slime GRPO Haiku training -- Modal launcher.

Three main entrypoints:
    modal run modal_train.py::prepare     # generate configs, download model, prepare data, deploy judges
    modal run modal_train.py::train       # kick off all training runs
    modal run modal_train.py::evaluate    # deploy model servers and run evals

Other utilities:
    modal run modal_train.py::list_configs
"""

import asyncio
import os
import shlex
import subprocess
import tempfile
import time

import modal
import modal.experimental

from configs import get_module, _CONFIGS_DIR
from configs.base import HF_CACHE_PATH, DATA_PATH, CHECKPOINTS_PATH, YAML_CONFIG_FIELDS

# -- Experiment (client-side only -- feeds decorator params)
experiment = os.environ.get("EXPERIMENT_CONFIG", "")
exp_mod = get_module(experiment) if experiment else None
modal_cfg = exp_mod.modal if exp_mod else None
slime_cfg = exp_mod.slime if exp_mod else None

# -- Image
slime_ROOT = "/root/slime"

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260329a")
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("llm_judges", copy=True)
    .add_local_python_source("config", copy=True)
)
if modal_cfg:
    for patch in modal_cfg.patch_files:
        image = image.add_local_file(
            patch, f"/tmp/{os.path.basename(patch)}", copy=True
        )
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)
    if modal_cfg.local_slime:
        image = image.add_local_dir(
            modal_cfg.local_slime,
            remote_path=slime_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

# -- Volumes
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

# -- App
app = modal.App(experiment or "train-haiku")

# -- All haiku experiment configs (base + judge variants)
ALL_HAIKU_CONFIGS = [
    "qwen3_4b_haiku",
    "qwen3_4b_haiku_standard_4b",
    "qwen3_4b_haiku_standard_30b",
    "qwen3_4b_haiku_standard_235b",
    "qwen3_4b_haiku_cl_4b",
    "qwen3_4b_haiku_cl_30b",
    "qwen3_4b_haiku_cl_235b",
]


# =============================================================================
# Remote functions
# =============================================================================


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * 60,
)
def _download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Download the model to the HF cache volume."""
    from huggingface_hub import snapshot_download

    slime_cfg = get_module(experiment).slime
    path = snapshot_download(repo_id=slime_cfg.hf_checkpoint)
    print(f"Model downloaded to {path}")
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * 60,
)
def _prepare_dataset(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run prepare_data() to populate the data volume."""
    slime_cfg = get_module(experiment).slime
    hf_cache_volume.reload()
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()


# -- Ray helpers
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def _start_ray_head(my_ip: str, n_nodes: int) -> None:
    """Start Ray head node and wait for all workers to join."""
    subprocess.Popen(
        ["ray", "start", "--head", f"--node-ip-address={my_ip}", "--dashboard-host=0.0.0.0"]
    )
    for _ in range(30):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Ray head node failed to start")

    for _ in range(60):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def _prepare_slime_cfg(slime_cfg, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs to temp files."""
    from huggingface_hub import snapshot_download
    import yaml

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(slime_cfg, attr, None)) and not str(val).startswith("/"):
            setattr(slime_cfg, attr, snapshot_download(val, local_files_only=True))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(slime_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} -> {path}")
            setattr(slime_cfg, field, path)


def _build_train_cmd(slime_cfg) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if slime_model_script is set."""
    train_script = (
        f"{slime_ROOT}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    )
    if slime_cfg.slime_model_script:
        inner = (
            f"source {slime_ROOT}/{slime_cfg.slime_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(slime_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(slime_cfg.cli_args())}"


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(slime_cfg.total_nodes(), rdma=True)
    if slime_cfg
    else lambda fn: fn
)
async def _train_single(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Single training run on a GPU cluster."""
    await asyncio.gather(hf_cache_volume.reload.aio(), data_volume.reload.aio())
    exp_mod = get_module(experiment)
    slime_cfg = exp_mod.slime
    modal_cfg = exp_mod.modal

    if slime_cfg.total_nodes() > 1:
        info = modal.experimental.get_cluster_info()
        rank, master_addr, my_ip = (
            info.rank,
            info.container_ipv4_ips[0],
            info.container_ipv4_ips[info.rank],
        )
        n_nodes = len(info.container_ipv4_ips)
    else:
        rank, master_addr, my_ip, n_nodes = 0, "127.0.0.1", "127.0.0.1", 1

    os.environ["slime_HOST_IP"] = my_ip

    if rank != 0:
        subprocess.Popen(
            ["ray", "start", f"--node-ip-address={my_ip}", "--address", f"{master_addr}:{RAY_PORT}"]
        )
        while True:
            await asyncio.sleep(10)

    _start_ray_head(my_ip, n_nodes)
    _prepare_slime_cfg(slime_cfg, tempfile.mkdtemp())

    if (wandb_key := os.environ.get("WANDB_API_KEY", "")) and getattr(
        slime_cfg, "use_wandb", False
    ):
        slime_cfg.wandb_key = wandb_key

    cmd = _build_train_cmd(slime_cfg)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **slime_cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    nodes = slime_cfg.total_nodes()
    gpu = f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}"
    mode = "async" if slime_cfg.async_mode else "sync"
    print(f"Job submitted: {job_id}")
    print(f"Training {experiment:<40} {nodes} node(s) x {gpu}  ({mode})")
    print(f"Command: {cmd}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)

    await checkpoints_volume.commit.aio()
    print("Checkpoints committed to volume")


# =============================================================================
# Local entrypoints
# =============================================================================


@app.local_entrypoint()
def list_configs():
    """Print all available experiments."""
    _skip = {"base", "__init__", "generate_judge_variants"}
    names = sorted(f.stem for f in _CONFIGS_DIR.glob("*.py") if f.stem not in _skip)
    print("Available experiments:")
    for name in names:
        mod = get_module(name)
        nodes = mod.slime.total_nodes()
        gpu = f"{mod.modal.gpu}:{mod.slime.actor_num_gpus_per_node}"
        mode = "async" if mod.slime.async_mode else "sync"
        print(f"  {name:<50} {nodes} node(s) x {gpu}  ({mode})")


@app.local_entrypoint()
def prepare():
    """Generate configs, download model, prepare dataset, and deploy LLM judges.

    Usage:
        modal run modal_train.py::prepare
    """
    import sys

    base_config = "qwen3_4b_haiku"

    # 1. Generate judge variant configs
    print("=== Generating judge variant configs ===")
    subprocess.run(
        [sys.executable, "configs/generate_judge_variants.py"],
        check=True,
    )

    # 2. Download model
    print("\n=== Downloading model ===")
    _download_model.remote(experiment=base_config)

    # 3. Prepare dataset
    print("\n=== Preparing dataset ===")
    _prepare_dataset.remote(experiment=base_config)

    # 4. Deploy LLM judges
    print("\n=== Deploying LLM judges ===")
    subprocess.run(["modal", "deploy", "llm_judges/deploy.py"], check=True)

    print("\n=== Prepare complete ===")


@app.local_entrypoint()
def train():
    """Kick off concurrent training runs for all haiku judge configurations.

    Usage:
        modal run modal_train.py::train
    """
    handles = []
    for config_name in ALL_HAIKU_CONFIGS:
        print(f"Spawning: {config_name}")
        handle = _train_single.spawn(experiment=config_name)
        handles.append((config_name, handle))

    print(f"\nSpawned {len(handles)} training runs. Waiting for completion...")
    for name, handle in handles:
        handle.get()
        print(f"Completed: {name}")


@app.local_entrypoint()
def evaluate():
    """Deploy model servers and run evals for all models.

    Usage:
        modal run modal_train.py::evaluate
    """
    from eval.shared import MODEL_CONFIG
    from eval.run_eval import run_eval

    # 1. Deploy model serving endpoints
    print("=== Deploying model servers ===")
    subprocess.run(["modal", "deploy", "eval/serve_haiku_model.py"], check=True)

    # 2. Run evals for each model
    print("\n=== Running evaluations ===")
    for model_key in MODEL_CONFIG:
        print(f"\n--- Evaluating: {model_key} ---")
        asyncio.run(run_eval(model_key=model_key))

    print("\n=== Evaluation complete ===")
