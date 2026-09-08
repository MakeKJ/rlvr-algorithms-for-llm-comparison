#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

# --- CONFIGURATION ---
# This path is relative to this script's location in environments/alfworld/
# It points to your virtual environment.
VENV_PATH="./.venv"

# --- SCRIPT START ---

# 1. Activate the virtual environment
echo "Activating Python virtual environment from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# 2. Navigate to the repository root
# The python commands must be run from the 'verl-agent' directory
# so that it can find the 'verl' and 'examples' modules.
echo "Changing directory to the verl-agent repository root..."
cd ../../source/verl-agent/

# --- PARAMETERS ---
# Reduced data sizes and training time for a quick verification run.
train_data_size=4  # 16 default
val_data_size=32  # 128 default
group_size=8 # Number of parallel environments
total_epochs=602
loss_mode="gspo"

# Set the engine for the model, default to vllm
ENGINE=${1:-vllm}
# vLLM will automatically use the best backend, likely FlashAttention on an H100.
# XFORMERS is a safe fallback.
export VLLM_ATTENTION_BACKEND=XFORMERS

# The CPU resource allocated for each environment worker.
num_cpus_per_env_worker=0.3

# 3. Data Preparation Step
echo "Preparing data..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# 4. Main Training Command
echo "Starting trainer main_ppo"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8  \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8  \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.05 \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop  \
    env.seed=1 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.webshop.use_small=False \
    trainer.resume_mode='auto' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gspo_qwen2.5_1.5b_run_1' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True $@ \
    trainer.max_actor_ckpt_to_keep=10

echo "Script finished successfully."