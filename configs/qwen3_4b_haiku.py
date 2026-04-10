"""Qwen3-4B GRPO on Haiku -- single node, colocated, structure-only reward (NO_LLM)."""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH

modal = ModalConfig(
    gpu="H200",
    image_run_commands=[
        "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",  # 4.54.1
        "uv pip install --system aiohttp nltk>=3.8.0",
        # Fix rope_theta access for transformers 5.x (moved to rope_parameters dict)
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
    ],
)


class _Slime(SlimeConfig):
    # -- Model -----------------------------------------------------------------
    hf_checkpoint = "Qwen/Qwen3-4B"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # -- Infrastructure --------------------------------------------------------
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # -- Data ------------------------------------------------------------------
    prompt_data = f"{DATA_PATH}/haiku/train.parquet"
    eval_prompt_data = ["haiku", f"{DATA_PATH}/haiku/test.parquet"]
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    apply_chat_template_kwargs = '{"enable_thinking": false}'
    rollout_shuffle = True

    # -- Reward model (structure-only by default) ------------------------------
    rm_type = "async_rm"
    custom_rm_path = "llm_judges.nlp.haiku_rm"

    # -- Rollout ---------------------------------------------------------------
    num_rollout = 50
    rollout_batch_size = 128
    rollout_max_response_len = 300
    rollout_temperature = 1
    rollout_skip_special_tokens = True
    rollout_num_gpus_per_engine = 2
    sglang_mem_fraction_static = 0.7
    n_samples_per_prompt = 8
    global_batch_size = 64

    # -- Eval ------------------------------------------------------------------
    eval_interval = 20
    n_samples_per_eval_prompt = 8
    eval_max_response_len = 300
    eval_top_p = 1

    # -- Training --------------------------------------------------------------
    tensor_model_parallel_size = 2
    sequence_parallel = True
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 9216
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    # -- Optimizer -------------------------------------------------------------
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # -- Algorithm -------------------------------------------------------------
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    # -- Model architecture (Qwen3-4B) ----------------------------------------
    num_layers = 36
    hidden_size = 2560
    ffn_hidden_size = 9728
    num_attention_heads = 32
    group_query_attention = True
    num_query_groups = 8
    kv_channels = 128
    vocab_size = 151936
    normalization = "RMSNorm"
    norm_epsilon = 1e-6
    swiglu = True
    disable_bias_linear = True
    qk_layernorm = True
    use_rotary_position_embeddings = True
    rotary_base = 1000000

    # -- Checkpointing ---------------------------------------------------------
    save = f"{CHECKPOINTS_PATH}/qwen3-4b-haiku"
    save_interval = 10

    # -- WandB -----------------------------------------------------------------
    use_wandb = True
    wandb_project = "example-train-haiku"
    wandb_group = "qwen3-4b-haiku"
    disable_wandb_random_suffix = True

    def prepare_data(self) -> None:
        """Download the statworx/haiku dataset, apply chat template, and save as parquet."""
        import os
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from llm_judges.base import MODAL_VOCABS

        tokenizer = AutoTokenizer.from_pretrained(self.hf_checkpoint)
        ds = load_dataset("statworx/haiku")

        vocab_str = ", ".join(MODAL_VOCABS)
        system_prompt = (
            "You are a haiku poet. You will be given a prompt and you will need to "
            "write a haiku about the prompt. Try to incorporate these words into the "
            f"haiku if possible: {vocab_str}"
        )

        def format_chat_template(example):
            keyword = example["keywords"].lower()
            question = f"Write me a haiku about {keyword}."
            messages = [
                {"content": system_prompt, "role": "system"},
                {"content": question, "role": "user"},
            ]
            return {
                "question": question,
                "label": example["text"],
                "messages": messages,
                "prompt": tokenizer.apply_chat_template(
                    messages, tokenize=False, enable_thinking=False
                ),
            }

        # Split: last 20% (max 1000) as test
        train_ds = ds["train"]
        test_size = min(1000, int(len(train_ds) * 0.2))
        test_ds = train_ds.select(range(len(train_ds) - test_size, len(train_ds)))
        train_ds = train_ds.select(range(len(train_ds) - test_size))

        train_transformed = train_ds.map(format_chat_template, remove_columns=["keywords"])
        test_transformed = test_ds.map(format_chat_template, remove_columns=["keywords"])

        os.makedirs(f"{DATA_PATH}/haiku", exist_ok=True)
        train_transformed.to_parquet(f"{DATA_PATH}/haiku/train.parquet")
        test_transformed.to_parquet(f"{DATA_PATH}/haiku/test.parquet")

        print(f"Train examples: {len(train_transformed)}")
        print(f"Test examples: {len(test_transformed)}")
        print(f"Example prompt: {train_transformed[0]['question']}")


slime = _Slime()
