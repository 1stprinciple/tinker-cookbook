"""Fireworks-specific runtime helpers for on-policy distillation.

Environment note:
This module expects the Fireworks training SDK to come from the installed
`fireworks-ai[training]` package in the active Python environment. A sibling
checkout such as `~/home/fireworks/` is not imported automatically just
because it exists on disk. Python only searches entries on `sys.path`; when
commands are run from `~/home/tinker-cookbook`, the local `tinker_cookbook`
package is found from the repo root, while `fireworks` resolves from the
active virtualenv's `site-packages` unless a local Fireworks source tree is
explicitly added to `PYTHONPATH`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import tinker
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks.utils.config import DeployConfig, InfraConfig
from tinker_cookbook.fireworks.utils.infra import setup_deployment
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import _training_logprobs_from_fwd_bwd
from tinker_cookbook.utils import trace
from tinker_cookbook.utils.misc_utils import safezip, split_list

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fireworks.training.sdk import DeploymentSampler, WeightSyncer
    from fireworks.training.sdk.client import FiretitanServiceClient, FiretitanTrainingClient


@dataclass(slots=True)
class FireworksDistillationRuntime:
    service_client: FiretitanServiceClient
    teacher_service_clients: list[FiretitanServiceClient]
    training_client: FiretitanTrainingClient
    teacher_clients: list[FiretitanTrainingClient]
    weight_syncer: WeightSyncer
    tokenizer: Any
    sampling_client: DeploymentSampler


def _validate_fireworks_config(
    *,
    base_url: str | None,
    fireworks_base_model_name: str | None,
) -> None:
    if base_url is None:
        raise ConfigurationError("Fireworks distillation requires 'base_url'.")
    if fireworks_base_model_name is None:
        raise ConfigurationError(
            "Fireworks distillation requires 'fireworks_base_model_name'."
        )


def _to_infra_config(cfg_section: DictConfig) -> InfraConfig:
    return InfraConfig(
        training_shape_id=cfg_section.get("training_shape_id"),
        ref_training_shape_id=cfg_section.get("ref_training_shape_id"),
        region=cfg_section.get("region"),
        custom_image_tag=cfg_section.get("custom_image_tag"),
        accelerator_type=cfg_section.get("accelerator_type"),
        accelerator_count=cfg_section.get("accelerator_count"),
        node_count=cfg_section.get("node_count", 1),
        extra_args=list(cfg_section.get("extra_args") or []),
    )


def _to_deploy_config(cfg_section: DictConfig) -> DeployConfig:
    return DeployConfig(
        deployment_id=cfg_section.get("deployment_id"),
        deployment_shape=cfg_section.get("deployment_shape"),
        deployment_region=cfg_section.get("deployment_region"),
        deployment_accelerator_type=cfg_section.get("deployment_accelerator_type"),
        hot_load_bucket_type=cfg_section.get("hot_load_bucket_type", "FW_HOSTED"),
        deployment_timeout_s=cfg_section.get("deployment_timeout_s", 5400),
        deployment_extra_args=list(cfg_section.get("deployment_extra_args") or []) or None,
        tokenizer_model=cfg_section.get("tokenizer_model"),
        sample_timeout=cfg_section.get("sample_timeout", 600),
        disable_speculative_decoding=cfg_section.get("disable_speculative_decoding", True),
        extra_values=dict(cfg_section.get("extra_values") or {}) or None,
    )


def _load_fireworks_defaults() -> tuple[str, InfraConfig, DeployConfig]:
    config_path = Path(__file__).with_name("fireworks.yaml")
    if not config_path.exists():
        raise ConfigurationError(f"Expected Fireworks defaults at {config_path}")

    cfg = OmegaConf.load(config_path)
    fireworks_base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")
    training_infra = cfg.get("training_infra")
    deployment = cfg.get("deployment")
    if training_infra is None or deployment is None:
        raise ConfigurationError(
            f"Fireworks defaults in {config_path} must define training_infra and deployment"
        )

    return (
        fireworks_base_url,
        _to_infra_config(training_infra),
        _to_deploy_config(deployment),
    )


def _create_deployment_manager(api_key: str):
    from fireworks.training.sdk import DeploymentManager

    fireworks_base_url, _, _ = _load_fireworks_defaults()
    return DeploymentManager(api_key=api_key, base_url=fireworks_base_url)


def _resolve_deployment_id(
    *,
    api_key: str,
    fireworks_base_model_name: str,
    fireworks_deployment_id: str | None,
) -> tuple[str, Any]:
    deploy_mgr = _create_deployment_manager(api_key)
    if fireworks_deployment_id is not None:
        return fireworks_deployment_id, deploy_mgr

    fireworks_base_url, infra, deploy = _load_fireworks_defaults()
    logger.info(
        "No fireworks_deployment_id provided; creating or attaching a deployment using %s",
        fireworks_base_url,
    )
    if infra.training_shape_id and not deploy.deployment_shape:
        from fireworks.training.sdk import TrainerJobManager

        trainer_mgr = TrainerJobManager(api_key=api_key, base_url=fireworks_base_url)
        profile = trainer_mgr.resolve_training_profile(infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(
            profile, "deployment_shape_version", None
        )
        if dep_shape:
            deploy.deployment_shape = dep_shape
            logger.info("Auto-derived deployment_shape from training shape: %s", dep_shape)

    deployment_info = setup_deployment(deploy_mgr, deploy, fireworks_base_model_name, infra)
    logger.info("Using Fireworks deployment %s", deployment_info.deployment_id)
    return deployment_info.deployment_id, deploy_mgr


def resolve_fireworks_tokenizer_model_name(fireworks_base_model_name: str) -> str:
    _, _, deploy = _load_fireworks_defaults()
    if deploy.tokenizer_model:
        return deploy.tokenizer_model
    if fireworks_base_model_name.count("/") == 1:
        return fireworks_base_model_name
    raise ConfigurationError(
        "Could not infer a Hugging Face tokenizer model for Fireworks distillation. "
        "Set deployment.tokenizer_model in tinker_cookbook/fireworks/fireworks.yaml."
    )


async def build_fireworks_distillation_runtime(
    *,
    base_url: str | None,
    model_name: str,
    lora_rank: int,
    renderer_name: str | None,
    load_checkpoint_path: str | None,
    dataset_configs: Sequence[DistillationDatasetConfig],
    start_batch: int,
    resume_state_path: str | None,
    user_metadata: dict[str, str],
    fireworks_base_model_name: str | None,
    fireworks_deployment_id: str | None,
    fireworks_hot_load_timeout: int,
    fireworks_dcp_timeout: int,
) -> FireworksDistillationRuntime:
    from fireworks.training.sdk import WeightSyncer
    from fireworks.training.sdk.client import FiretitanServiceClient
    from fireworks.training.sdk import DeploymentSampler

    _validate_fireworks_config(
        base_url=base_url,
        fireworks_base_model_name=fireworks_base_model_name,
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    service_client = FiretitanServiceClient(base_url=base_url, api_key=api_key)

    training_client = service_client.create_training_client(
        base_model=model_name,
        lora_rank=lora_rank,
    )
    if resume_state_path is not None:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_state_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_state_path, user_metadata=user_metadata
        )
        logger.info("Resumed training from %s", resume_state_path)
    elif load_checkpoint_path is not None:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, load_checkpoint_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info("Loaded weights from %s", load_checkpoint_path)
        training_client.load_state(load_checkpoint_path)

    deployment_id, deploy_mgr = _resolve_deployment_id(
        api_key=api_key,
        fireworks_base_model_name=cast(str, fireworks_base_model_name),
        fireworks_deployment_id=fireworks_deployment_id,
    )
    weight_syncer = WeightSyncer(
        policy_client=training_client,
        deploy_mgr=deploy_mgr,
        deployment_id=deployment_id,
        base_model=fireworks_base_model_name,
        hotload_timeout=fireworks_hot_load_timeout,
    )
    checkpoint_name = f"resume-{start_batch}-base" if start_batch > 0 else "step-0-base"
    weight_syncer.save_and_hotload(checkpoint_name, checkpoint_type="base")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_fireworks_tokenizer_model_name(cast(str, fireworks_base_model_name)),
        trust_remote_code=True,
    )
    sampling_client = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=f"accounts/{deploy_mgr.account_id}/deployments/{deployment_id}",
        api_key=api_key,
        tokenizer=tokenizer,
    )

    teacher_service_clients: list[FiretitanServiceClient] = []
    teacher_clients_by_key: dict[tuple[str, str | None], FiretitanTrainingClient] = {}
    teacher_clients: list[FiretitanTrainingClient] = []
    for dataset_config in dataset_configs:
        teacher_config = dataset_config.teacher_config
        teacher_key = (teacher_config.base_model, teacher_config.load_checkpoint_path)
        teacher_client = teacher_clients_by_key.get(teacher_key)
        if teacher_client is None:
            teacher_service_client = FiretitanServiceClient(base_url=base_url, api_key=api_key)
            teacher_service_clients.append(teacher_service_client)
            teacher_client = teacher_service_client.create_training_client(
                base_model=teacher_config.base_model,
                lora_rank=lora_rank,
            )
            if teacher_config.load_checkpoint_path is not None:
                teacher_client.load_state(teacher_config.load_checkpoint_path)
            teacher_clients_by_key[teacher_key] = teacher_client
        teacher_clients.append(teacher_client)
        logger.info(
            "Created teacher training client for %s (checkpoint: %s)",
            teacher_config.base_model,
            teacher_config.load_checkpoint_path,
        )

    return FireworksDistillationRuntime(
        service_client=service_client,
        teacher_service_clients=teacher_service_clients,
        training_client=training_client,
        teacher_clients=teacher_clients,
        weight_syncer=weight_syncer,
        tokenizer=tokenizer,
        sampling_client=sampling_client,
    )


@trace.scope
async def get_initial_sampling_client(
    runtime: FireworksDistillationRuntime,
    *,
    start_batch: int,
    log_path: str,
    save_every: int,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    return runtime.sampling_client, {}


@trace.scope
async def train_step_with_fireworks_client(
    *,
    data_D: list[tinker.Datum],
    training_client: FiretitanTrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: str,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []
    optim_result = None

    for i, batch in enumerate(batches):
        logger.info(
            "[fireworks_train_step] submitting forward_backward: substep=%d batch_size=%d",
            i,
            len(batch),
        )
        fwd_bwd_future = await training_client.forward_backward_async(
            [
                tinker.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
                )
                for datum in batch
            ],
            loss_fn,
            loss_fn_config,
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()
        logger.info("[fireworks_train_step] received forward_backward result: substep=%d", i)
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))

        logger.info("[fireworks_train_step] submitting optim_step: substep=%d", i)
        optim_future = await training_client.optim_step_async(adam_params)
        optim_result = await optim_future.result_async()
        logger.info("[fireworks_train_step] received optim_step result: substep=%d", i)

    if metrics is not None and optim_result is not None and optim_result.metrics:
        metrics.update(optim_result.metrics)

    return training_logprobs_D


@trace.scope
async def refresh_sampling_client_after_train_step(
    runtime: FireworksDistillationRuntime,
    *,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    log_path: str,
    save_every: int,
    compute_post_kl: bool,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    from tinker_cookbook.rl.metrics import compute_kl_sample_train, compute_post_kl

    metrics = compute_kl_sample_train(data_D, training_logprobs_D)
    logger.info(
        "Saving checkpoint after batch %d (save_every=%d, compute_post_kl=%s)",
        i_batch,
        save_every,
        compute_post_kl,
    )
    if save_every > 0 and i_batch % save_every == 0:
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=runtime.training_client,
            name=f"{i_batch:06d}",
            log_path=log_path,
            loop_state={"batch": i_batch},
            kind="both",
            ttl_seconds=None,
        )
    else:
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=runtime.training_client,
            name=f"{i_batch:06d}",
            log_path=log_path,
            loop_state={"batch": i_batch},
            kind="sampler",
            ttl_seconds=None,
        )
    logger.info("Saved checkpoint artifacts after batch %d: %s", i_batch, sorted(path_dict.keys()))

    logger.info("Hotloading sampler checkpoint after batch %d", i_batch)
    success = runtime.weight_syncer.hotload(path_dict["sampler_path"], checkpoint_type="delta")
    if not success:
        raise RuntimeError(f"Failed to hotload checkpoint {path_dict['sampler_path']}")
    logger.info("Hotload complete after batch %d", i_batch)

    if compute_post_kl:
        logger.info("Computing post-update KL after batch %d", i_batch)
        metrics.update(await compute_post_kl(data_D, runtime.sampling_client))
        logger.info("Computed post-update KL after batch %d", i_batch)

    return runtime.sampling_client, metrics


@trace.scope
async def incorporate_teacher_kl_penalty(
    data_D: list[tinker.Datum],
    teacher_clients_D: list[FiretitanTrainingClient],
    dataset_indices_D: list[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    async def _forward_teacher_logprobs(
        teacher_client: FiretitanTrainingClient, datum: tinker.Datum, datum_idx: int
    ) -> torch.Tensor:
        logger.info("Submitting teacher forward for datum %d", datum_idx)
        future = await teacher_client.forward_async(
            [
                tinker.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
                )
            ],
            "cross_entropy",
        )
        result = await future.result_async()
        logger.info("Teacher forward completed for datum %d", datum_idx)
        return result.loss_fn_outputs[0]["logprobs"].to_torch()

    logger.info("Dispatching %d teacher forward requests", len(data_D))
    teacher_logprobs_D = await asyncio.gather(
        *[
            _forward_teacher_logprobs(teacher_client, datum, i)
            for i, (teacher_client, datum) in enumerate(zip(teacher_clients_D, data_D))
        ]
    )
    logger.info("Collected %d teacher forward results", len(teacher_logprobs_D))
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = [
        (sampled_logprobs - teacher_logprobs) * mask
        for teacher_logprobs, sampled_logprobs, mask in safezip(
            teacher_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    per_dataset_kl: dict[int, tuple[float, float]] = {}

    for i, datum in enumerate(data_D):
        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = float_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    metrics = {"teacher_kl": float(avg_logp_diff)}
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    return metrics
