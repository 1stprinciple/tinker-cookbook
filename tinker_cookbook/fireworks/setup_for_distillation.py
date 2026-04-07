import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig

from fireworks.training.sdk import (
    DeploymentManager,
    DeploymentSampler,
    TrainerJobManager,
    WeightSyncer,
)
from tinker_cookbook.fireworks.utils import ReconnectableClient, create_trainer_job, setup_deployment
from tinker_cookbook.fireworks.utils.config import InfraConfig, DeployConfig
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _to_infra_config(cfg_section: DictConfig) -> InfraConfig:
    """Convert an OmegaConf ``training_infra`` section to an ``InfraConfig`` dataclass."""
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
    """Convert an OmegaConf ``deployment`` section to a ``DeployConfig`` dataclass."""
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


def init_fireworks_distillation_infra(cfg: DictConfig) -> tuple:
    """Create Fireworks infra for on-policy distillation.

    Sets up:
    - Student trainer job + ReconnectableClient + WeightSyncer
    - DeploymentSampler for student sampling (via hotloaded deployment)
    - Teacher sampling client (via the same deployment or a separate model)

    Returns:
        (student_endpoint, sampling_client, weight_syncer, teacher_model_name)
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    infra = _to_infra_config(cfg.training_infra)
    deploy = _to_deploy_config(cfg.deployment)

    # Resolve training shape profile and auto-derive config values
    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(profile, "deployment_shape_version", None)
        if dep_shape and not deploy.deployment_shape:
            deploy.deployment_shape = dep_shape
            logger.info("Auto-derived deployment_shape from training shape: %s", dep_shape)
        if profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = profile.max_supported_context_length
            logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)

    dep_info = setup_deployment(deploy_mgr, deploy, cfg.model.name, infra)
    deployment_id = dep_info.deployment_id

    # Create student trainer job
    student_ep = create_trainer_job(
        rlor_mgr,
        base_model=cfg.model.name,
        infra=infra,
        profile=profile,
        lora_rank=cfg.model.get("lora_rank", 0),
        max_seq_len=cfg.training.max_length,
        learning_rate=cfg.training.learning_rate,
        display_name=cfg.get("display_name", "distill-student"),
        job_id=cfg.training_infra.training_job_id,
        hot_load_deployment_id=deployment_id,
    )

    student_rc = ReconnectableClient(
        rlor_mgr, student_ep.job_id, cfg.model.name,
        lora_rank=cfg.model.get("lora_rank", 0),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        deploy.tokenizer_model or cfg.model.name,
        trust_remote_code=True,
    )
    inference_model = dep_info.inference_model if dep_info else cfg.model.name
    sampling_client = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
    )
    weight_syncer = WeightSyncer(
        policy_client=student_rc.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=deployment_id,
        base_model=cfg.model.name,
        hotload_timeout=cfg.hotload.hot_load_timeout,
    )

    # Teacher model name from config (teacher sampling is handled by
    # tinker_cookbook.distillation via FiretitanServiceClient.create_sampling_client)
    teacher_model = cfg.teacher.name
    teacher_checkpoint = cfg.teacher.get("checkpoint_path")

    return student_ep, sampling_client, weight_syncer, teacher_model, teacher_checkpoint


@hydra.main(config_path=".", config_name="fireworks_distillation", version_base=None)
def main(cfg: DictConfig) -> None:
    student_ep, sampling_client, weight_syncer, teacher_model, teacher_checkpoint = (
        init_fireworks_distillation_infra(cfg)
    )
    logger.info("Fireworks student endpoint ready (student=%s)", student_ep.base_url)
    logger.info("Fireworks sampling client ready (sampling_client=%s)", sampling_client.model)
    logger.info("Teacher model: %s (checkpoint: %s)", teacher_model, teacher_checkpoint)


if __name__ == "__main__":
    main()
