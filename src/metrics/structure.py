"""Structural (edge-based) evaluation metrics (AGENT.md §10)."""

import torch
import torch.nn.functional as F
from torch import Tensor

_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]


def edge_map(img: Tensor, percentile: float = 99.0) -> Tensor:
    """Binary Sobel edge map with shape (N, 1, H, W) and values in {0.0, 1.0}.

    Gradient magnitude from fixed 3x3 Sobel kernels with 1-pixel replicate padding
    (so a constant image has exactly zero gradient everywhere, borders included);
    each image is thresholded at `percentile` of its own magnitude map, and values
    strictly greater than the threshold count as edges. Images with zero gradient
    everywhere yield empty edge maps.
    """
    kernel_x = torch.tensor(_SOBEL_X, dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(img, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    magnitude = torch.sqrt(gx * gx + gy * gy)
    flat = magnitude.reshape(magnitude.shape[0], -1)
    thresh = torch.quantile(flat, percentile / 100.0, dim=1, keepdim=True)
    return (flat > thresh).reshape(img.shape).float()


def structural_precision_recall(
    pred: Tensor,
    target: Tensor,
    tolerance_px: int = 2,
    percentile: float = 99.0,
) -> dict[str, float]:
    """Edge-based structural precision / recall / F1 (AGENT.md §10), batch-averaged.

    Edge maps of SR and GT are dilated with a (2*tolerance_px+1) max-pool window;
    tolerance-based matching gives:
        precision = |E_SR and dilate(E_GT)| / max(|E_SR|, 1)
        recall    = |E_GT and dilate(E_SR)| / max(|E_GT|, 1)
        f1        = 2PR / (P + R), 0 when P + R == 0.
    Images whose edge maps are both empty contribute 0.0 to all three metrics.
    """
    edges_pred = edge_map(pred, percentile)
    edges_target = edge_map(target, percentile)
    kernel = 2 * tolerance_px + 1
    dilated_pred = F.max_pool2d(edges_pred, kernel_size=kernel, stride=1, padding=tolerance_px)
    dilated_target = F.max_pool2d(edges_target, kernel_size=kernel, stride=1, padding=tolerance_px)

    count_pred = edges_pred.sum(dim=(1, 2, 3))
    count_target = edges_target.sum(dim=(1, 2, 3))
    both_empty = (count_pred == 0) & (count_target == 0)

    precision = (edges_pred * dilated_target).sum(dim=(1, 2, 3)) / count_pred.clamp(min=1.0)
    recall = (edges_target * dilated_pred).sum(dim=(1, 2, 3)) / count_target.clamp(min=1.0)
    precision = torch.where(both_empty, torch.zeros_like(precision), precision)
    recall = torch.where(both_empty, torch.zeros_like(recall), recall)
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12), torch.zeros_like(denom))
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }
