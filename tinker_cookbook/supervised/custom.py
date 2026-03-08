"""Custom loss functions for forward_backward_custom.

These are not used by the main training loop (which uses the built-in
"cross_entropy" loss), but are available for experimentation.
"""

import tinker
import torch

from tinker_cookbook.supervised.common import compute_mean_nll


def custom_cross_entropy(
    data: list[tinker.Datum], logprobs: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy loss equivalent to Tinker's built-in ``cross_entropy``."""
    total_loss = torch.tensor(0.0, device=logprobs[0].device, requires_grad=True)

    for datum, target_logprobs in zip(data, logprobs):
        weights = datum.loss_fn_inputs["weights"].to_torch().to(target_logprobs.device)
        total_loss = total_loss + (-target_logprobs * weights).sum()

    logprobs_td = [tinker.TensorData.from_torch(lp.detach()) for lp in logprobs]
    weights_td = [datum.loss_fn_inputs["weights"] for datum in data]
    train_mean_nll = compute_mean_nll(logprobs_td, weights_td)

    return total_loss, {"train_mean_nll": train_mean_nll}
