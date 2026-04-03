"""
Implements on-policy distillation. For more details, see:
https://thinkingmachines.ai/blog/on-policy-distillation
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
import tinker
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks.distillation_runtime import (
    FireworksDistillationRuntime,
    build_fireworks_distillation_runtime,
    get_initial_sampling_client,
    incorporate_teacher_kl_penalty,
    refresh_sampling_client_after_train_step,
    train_step_with_fireworks_client,
)
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.train import (
    do_group_rollout_and_filter_constant_reward,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.deprecation import warn_deprecated
from tinker_cookbook.utils.misc_utils import iteration_dir

logger = logging.getLogger(__name__)


@trace.scope
async def incorporate_kl_penalty(
    data_D: list[tinker.Datum],
    teacher_clients_D: list[Any],
    dataset_indices_D: list[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    return await incorporate_teacher_kl_penalty(
        data_D,
        teacher_clients_D,
        dataset_indices_D,
        kl_penalty_coef,
        kl_discount_factor,
    )


@chz.chz
class Config:
    learning_rate: float
    dataset_configs: list[DistillationDatasetConfig]
    model_name: str
    renderer_name: str | None = None
    max_tokens: int
    temperature: float = 1.0
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    base_url: str | None = None
    enable_trace: bool = False
    span_chart_every: int = 0

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    # Maximum number of training steps. If None, train on the full dataset.
    max_steps: int | None = None
    # Deprecated alias for max_steps. Use max_steps instead.
    max_step: int | None = None

    fireworks_base_model_name: str | None = None
    fireworks_deployment_id: str | None = None
    fireworks_hot_load_timeout: int = 600
    fireworks_dcp_timeout: int = 2700


@trace.scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: list[int],
    teacher_clients: list[Any],
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""
    logger.info(
        "Preparing minibatch: groups=%d trajectory_groups=%d",
        len(env_group_builders_P),
        len(trajectory_groups_P),
    )

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    async with trace.scope_span("assemble_training_data"):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)
    logger.info("Prepared training data: datums=%d", len(data_D))

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        logger.info("Starting teacher KL penalty computation: datums=%d", len(data_D))
        async with trace.scope_span("compute_kl_penalty"):
            # Map each datum to its teacher sampling client and dataset index using metadata
            #   - metadata_D contains group_idx which indexes into trajectory_groups_P
            #   - dataset_indices_P[group_idx] gives us the dataset index
            #   - teacher_clients[dataset_idx] gives us the teacher
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                teacher_clients_D,
                dataset_indices_D,
                kl_penalty_coef,
                kl_discount_factor,
            )
        logger.info("Completed teacher KL penalty computation")
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@trace.scope
async def do_train_step_and_get_sampling_client(
    config: Config,
    i_batch: int,
    fireworks_runtime: FireworksDistillationRuntime,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: list[int],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    trace.update_scope_context({"step": i_batch})
    logger.info("Starting train step for batch %d", i_batch)

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        fireworks_runtime.teacher_clients,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    async with trace.scope_span("train"):
        logger.info("Submitting trainer forward/backward for batch %d", i_batch)
        training_logprobs_D = await train_step_with_fireworks_client(
            data_D=data_D,
            training_client=fireworks_runtime.training_client,
            learning_rate=config.learning_rate,
            num_substeps=config.num_substeps,
            loss_fn=config.loss_fn,
            loss_fn_config=config.loss_fn_config,
            metrics=metrics,
        )
    logger.info(
        "Trainer forward/backward completed for batch %d with %d logprob tensors",
        i_batch,
        len(training_logprobs_D),
    )

    logger.info("Refreshing sampling client after batch %d", i_batch)
    sampling_client, full_batch_metrics = await refresh_sampling_client_after_train_step(
        fireworks_runtime,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch=i_batch + 1,
        data_D=data_D,
        training_logprobs_D=training_logprobs_D,
        log_path=config.log_path,
        save_every=config.save_every,
        compute_post_kl=config.compute_post_kl,
    )
    metrics.update(full_batch_metrics)
    logger.info("Completed train step for batch %d", i_batch)

    return sampling_client, metrics


@trace.scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    fireworks_runtime: FireworksDistillationRuntime,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await get_initial_sampling_client(
        fireworks_runtime,
        start_batch=start_batch,
        log_path=config.log_path,
        save_every=config.save_every,
    )

    log_path = Path(config.log_path)

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Run evaluations
            if config.eval_every > 0 and i_batch % config.eval_every == 0:
                async with trace.scope_span("run_evals"):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

            # Get batch and sample trajectories
            env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
            async with trace.scope_span("sample"):
                logger.info(
                    "Starting rollout sampling for batch %d with %d env groups",
                    i_batch,
                    len(env_group_builders_P),
                )
                trajectory_groups_P = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                temperature=config.temperature,
                                max_tokens=config.max_tokens,
                                do_remove_constant_reward_groups=False,
                            ),
                            name=f"sample_task_{i}",
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ],
                )
            logger.info("Completed rollout sampling for batch %d", i_batch)
            trajectory_groups_P = [
                trajectory_group
                for trajectory_group in trajectory_groups_P
                if trajectory_group is not None
            ]
            logger.info(
                "Retained %d trajectory groups after filtering for batch %d",
                len(trajectory_groups_P),
                i_batch,
            )

            # Train step
            sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                config,
                i_batch,
                fireworks_runtime,
                tokenizer,
                env_group_builders_P,
                trajectory_groups_P,
                dataset_indices_P,
            )

            metrics.update(train_step_metrics)

        # Log timing metrics from trace_iteration window
        metrics.update(window.get_timing_metrics())
        window.write_spans_jsonl(log_path / "timing_spans.jsonl", step=i_batch)
        if config.span_chart_every > 0 and i_batch % config.span_chart_every == 0:
            iter_dir = iteration_dir(log_path, i_batch)
            if iter_dir is not None:
                iter_dir.mkdir(parents=True, exist_ok=True)
                trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
        ml_logger.log_metrics(metrics, step=i_batch)


@trace.scope
async def main(
    config: Config | None = None,
    *,
    cfg: Config | None = None,
):
    """Main training loop for on-policy distillation."""
    if cfg is not None:
        warn_deprecated("cfg", removal_version="0.3.0", message="Use 'config' instead.")
        if config is not None:
            raise ConfigurationError("Cannot pass both 'config' and 'cfg'. Use 'config'.")
        config = cfg
    if config is None:
        raise ConfigurationError("'config' is required.")

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    if config.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace.trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_batch = resume_info.batch
    else:
        start_batch = 0

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    fireworks_runtime = await build_fireworks_distillation_runtime(
        base_url=config.base_url,
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        renderer_name=config.renderer_name,
        load_checkpoint_path=config.load_checkpoint_path,
        dataset_configs=config.dataset_configs,
        start_batch=start_batch,
        resume_state_path=resume_info.state_path if resume_info else None,
        user_metadata=user_metadata,
        fireworks_base_model_name=config.fireworks_base_model_name,
        fireworks_deployment_id=config.fireworks_deployment_id,
        fireworks_hot_load_timeout=config.fireworks_hot_load_timeout,
        fireworks_dcp_timeout=config.fireworks_dcp_timeout,
    )

    # Fireworks model ids are not always valid Hugging Face tokenizer ids,
    # so the runtime resolves an explicit tokenizer model when needed.
    tokenizer = fireworks_runtime.tokenizer

    # Create datasets and teacher clients from configs
    datasets = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in config.evaluator_builders]

    for dataset_config in config.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=config.max_tokens))

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    # Resolve max_steps from either max_steps or deprecated max_step
    effective_max_steps = config.max_steps
    if config.max_step is not None:
        if config.max_steps is not None:
            raise ConfigurationError("Cannot specify both max_steps and max_step. Use max_steps.")
        warn_deprecated("max_step", removal_version="0.3.0", message="Use 'max_steps' instead.")
        effective_max_steps = config.max_step
    num_batches = (
        min(effective_max_steps, num_batches) if effective_max_steps is not None else num_batches
    )
    logger.info(f"Will train on {num_batches} batches (dataset has {num_batches})")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        config=config,
        fireworks_runtime=fireworks_runtime,
        evaluators=evaluators,
        dataset=composite_dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=fireworks_runtime.training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=None,
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
