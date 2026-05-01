#!/usr/bin/env python3
"""Promote a sampler checkpoint to a deployable Fireworks model.

Unlike the cookbook variant, the ``checkpoints.jsonl`` written by this
package does not record the source trainer job ID, so it must be passed
on the command line along with the sampler path, base model, and output
model ID.

Usage:
    export FIREWORKS_API_KEY=...

    python -m tinker_cookbook.fireworks.promote_checkpoint \\
        --source-job-id <trainer-job-id> \\
        --sampler-path <sampler-checkpoint-name> \\
        --output-model-id my-fine-tuned-model \\
        --base-model accounts/fireworks/models/qwen3-8b
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import cast

from dotenv import load_dotenv
from fireworks.training.sdk import TrainerJobManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a sampler checkpoint to a deployable Fireworks model",
    )
    parser.add_argument(
        "--source-job-id",
        required=True,
        help="Trainer job ID that produced the sampler checkpoint.",
    )
    parser.add_argument(
        "--sampler-path",
        required=True,
        help="Sampler checkpoint name/path (e.g. the snapshot name returned by save_weights_for_sampler_ext).",
    )
    parser.add_argument(
        "--output-model-id",
        required=True,
        help="Promoted model ID.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model for metadata inheritance, e.g. accounts/fireworks/models/qwen3-8b.",
    )
    parser.add_argument(
        "--hot-load-deployment-id",
        default=None,
        help=(
            "[Legacy] Deployment ID for jobs from deployments that predate "
            "the stored-bucket-URL migration. Modern runs do not need this."
        ),
    )
    return parser.parse_args()


def promote_checkpoint(
    *,
    source_job_id: str,
    sampler_path: str,
    output_model_id: str,
    base_model: str,
    hot_load_deployment_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, object]:
    api_key = api_key or os.environ["FIREWORKS_API_KEY"]
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    client = TrainerJobManager(api_key=api_key, base_url=base_url)

    logger.info("Checkpoint:      %s", sampler_path)
    logger.info("Source job:      %s", source_job_id)
    logger.info("Base model:      %s", base_model)
    logger.info("Output model ID: %s", output_model_id)
    if hot_load_deployment_id:
        logger.info("Deployment ID:   %s (legacy)", hot_load_deployment_id)

    model = client.promote_checkpoint(
        source_job_id,
        sampler_path,
        output_model_id,
        base_model,
        hot_load_deployment_id=hot_load_deployment_id,
    )

    logger.info(
        "Promoted model: %s",
        model.get("name", f"accounts/{client.account_id}/models/{output_model_id}"),
    )
    logger.info(
        "Model state=%s kind=%s",
        model.get("state", "UNKNOWN"),
        model.get("kind", "UNKNOWN"),
    )
    return cast(dict[str, object], model)


def main() -> None:
    args = parse_args()
    promote_checkpoint(
        source_job_id=args.source_job_id,
        sampler_path=args.sampler_path,
        output_model_id=args.output_model_id,
        base_model=args.base_model,
        hot_load_deployment_id=args.hot_load_deployment_id,
    )


if __name__ == "__main__":
    main()
