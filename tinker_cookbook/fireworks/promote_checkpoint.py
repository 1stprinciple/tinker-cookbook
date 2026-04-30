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


def main() -> None:
    args = parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    client = TrainerJobManager(api_key=api_key, base_url=base_url)

    logger.info("Checkpoint:      %s", args.sampler_path)
    logger.info("Source job:      %s", args.source_job_id)
    logger.info("Base model:      %s", args.base_model)
    logger.info("Output model ID: %s", args.output_model_id)
    if args.hot_load_deployment_id:
        logger.info("Deployment ID:   %s (legacy)", args.hot_load_deployment_id)

    model = client.promote_checkpoint(
        args.source_job_id,
        args.sampler_path,
        args.output_model_id,
        args.base_model,
        hot_load_deployment_id=args.hot_load_deployment_id,
    )

    logger.info(
        "Promoted model: %s",
        model.get("name", f"accounts/{client.account_id}/models/{args.output_model_id}"),
    )
    logger.info(
        "Model state=%s kind=%s",
        model.get("state", "UNKNOWN"),
        model.get("kind", "UNKNOWN"),
    )


if __name__ == "__main__":
    main()
