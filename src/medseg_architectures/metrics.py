"""Segmentation metrics for binary masks and integer class maps.

Segmentation evaluation needs more than accuracy because medical images are
usually dominated by background. A background-only prediction can look accurate
pixel by pixel while completely missing the lesion, organ, or boundary that the
experiment is meant to measure.

The metrics in this module fall into two families. Dice, IoU, sensitivity, and
specificity are overlap or confusion-count metrics: they compare which pixels or
voxels belong to the predicted and target regions. HD95 is a boundary metric: it
compares distances between mask surfaces, which matters when a small boundary
shift has clinical consequences.

All public functions expect binary masks or integer class maps, not raw logits.
Callers should apply ``sigmoid`` plus thresholding for binary segmentation or
``argmax`` for multiclass segmentation before using these metrics.

Quick reference: ``ClassMetrics`` and ``SegmentationMetrics`` hold results;
``dice_score``, ``iou_score``, ``sensitivity``, ``specificity``, and
``hausdorff_95`` compute individual metrics; ``evaluate`` computes the standard
set together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ClassMetrics:
    """Overlap and boundary scores for a single segmentation class."""

    dice: float
    iou: float
    sensitivity: float
    specificity: float
    hausdorff_95: float | None


@dataclass(frozen=True)
class SegmentationMetrics:
    """Aggregate evaluation result across all foreground classes.

    For binary segmentation, ``per_class`` contains one entry for the foreground
    mask. For multiclass segmentation with class labels ``0`` through ``K - 1``,
    ``per_class`` contains one entry for each foreground label ``1`` through
    ``K - 1``; ``per_class[0]`` therefore corresponds to class label ``1``.
    Background is excluded from ``per_class`` and from foreground macro-averaged
    Dice and IoU, but it contributes to the true-negative and false-positive
    counts used by one-vs-rest specificity.
    """

    mean_dice: float
    mean_iou: float
    mean_sensitivity: float
    mean_specificity: float
    mean_hausdorff_95: float | None
    per_class: tuple[ClassMetrics, ...]


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 1,
    smooth: float = 1e-6,
) -> float:
    """Measure overlap between predicted and target foreground regions.

    Dice exists because pixel accuracy can be dominated by easy background. In a
    typical MRI liver segmentation task, a model that predicts all background
    can score above 99% accuracy when the liver occupies a small fraction of the
    image, while Dice is 0 because no liver was found. Clinically, Dice ranges
    from 0 for no overlap to 1 for perfect overlap; a Dice of 0.85 on a liver
    segmentation is usually a strong benchmark result, while the same value on a
    tiny lesion may still hide a clinically important miss. Dice can mislead on
    very small structures because the ``smooth`` term and a handful of pixels can
    dominate the score.

    Formula:
    ``Dice = (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)``.
    The ``2`` normalizes the intersection to the mean of the predicted and
    target mask sizes: if prediction and target are identical, the numerator and
    denominator match. ``smooth`` prevents a ``0/0`` denominator when both masks
    are empty. Under the standard empty-class convention, an empty prediction and
    empty target score 1.0, which matters for rare structures that may be absent
    in valid cases.

    Args:
        pred: Boolean or integer tensor of predicted labels shaped ``(H, W)`` or
            ``(D, H, W)`` for one sample.
        target: Boolean or integer tensor of ground-truth labels with the same
            shape as ``pred``.
        num_classes: Use ``1`` for binary masks or ``K`` for multiclass labels
            ``0`` through ``K - 1``. Multiclass Dice averages foreground labels
            only, excluding background label ``0``.
        smooth: Additive smoothing constant used to define empty-mask behavior.

    Returns:
        Mean foreground Dice as a Python ``float``.
    """

    pred_cpu, target_cpu = _validate_metric_inputs(
        pred,
        target,
        num_classes=num_classes,
        smooth=smooth,
    )
    return float(
        np.mean(
            [
                _dice_for_masks(pred_mask, target_mask, smooth)
                for pred_mask, target_mask in _foreground_masks(
                    pred_cpu,
                    target_cpu,
                    num_classes,
                )
            ]
        )
    )


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 1,
    smooth: float = 1e-6,
) -> float:
    """Measure intersection over union between prediction and target.

    IoU exists to penalize false-positive and false-negative area using the
    union as the reference set. It addresses the same class-imbalance failure as
    Dice but is stricter for imperfect overlaps. Clinically, IoU ranges from 0
    to 1; an IoU of 0.80 means most of the combined predicted-or-target region
    agrees, but the remaining 20% may still matter near a treatment boundary.
    IoU can mislead when users treat it as independent evidence from Dice,
    because the two metrics always rank the same masks the same way.

    Formula:
    ``IoU = (|P ∩ T| + smooth) / (|P ∪ T| + smooth)``.
    Dice and IoU are algebraically linked by
    ``IoU = Dice / (2 - Dice)``, so choosing between them is usually about
    reporting convention. Prefer IoU when comparing to computer vision
    benchmarks such as COCO or when false positives need stricter penalization.
    Prefer Dice for most medical segmentation papers because reviewers expect
    it as the primary overlap metric.

    Args:
        pred: Boolean or integer tensor of predicted labels shaped ``(H, W)`` or
            ``(D, H, W)`` for one sample.
        target: Boolean or integer tensor of ground-truth labels with the same
            shape as ``pred``.
        num_classes: Use ``1`` for binary masks or ``K`` for multiclass labels
            ``0`` through ``K - 1``. Multiclass IoU averages foreground labels
            only, excluding background label ``0``.
        smooth: Additive smoothing constant used to define empty-mask behavior.

    Returns:
        Mean foreground IoU as a Python ``float``.
    """

    pred_cpu, target_cpu = _validate_metric_inputs(
        pred,
        target,
        num_classes=num_classes,
        smooth=smooth,
    )
    return float(
        np.mean(
            [
                _iou_for_masks(pred_mask, target_mask, smooth)
                for pred_mask, target_mask in _foreground_masks(
                    pred_cpu,
                    target_cpu,
                    num_classes,
                )
            ]
        )
    )


def sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 1,
    smooth: float = 1e-6,
) -> float:
    """Measure what fraction of true foreground was captured.

    Sensitivity, also called recall, exists because overlap scores alone can
    hide whether the dominant error is missing real foreground. Clinically, a
    low sensitivity means the model misses real lesions; in a screening setting,
    missing a tumor is often the primary error to avoid. Sensitivity ranges from
    0 to 1, where 1 means every target foreground pixel or voxel was found.
    Sensitivity can mislead under severe class imbalance because it ignores true
    negatives and does not say how much healthy tissue was incorrectly flagged.

    Formula:
    ``Sensitivity = TP / (TP + FN + smooth)``.
    Sensitivity is asymmetric with specificity. Raising or lowering the
    probability threshold changes both metrics, so threshold selection belongs
    to the evaluation setup rather than the architecture design.

    Args:
        pred: Boolean or integer tensor of predicted labels shaped ``(H, W)`` or
            ``(D, H, W)`` for one sample.
        target: Boolean or integer tensor of ground-truth labels with the same
            shape as ``pred``.
        num_classes: Use ``1`` for binary masks or ``K`` for multiclass labels
            ``0`` through ``K - 1``. Multiclass sensitivity averages foreground
            labels only.
        smooth: Additive smoothing constant for stable denominators.

    Returns:
        Mean foreground sensitivity as a Python ``float``.
    """

    pred_cpu, target_cpu = _validate_metric_inputs(
        pred,
        target,
        num_classes=num_classes,
        smooth=smooth,
    )
    return float(
        np.mean(
            [
                _sensitivity_for_masks(pred_mask, target_mask, smooth)
                for pred_mask, target_mask in _foreground_masks(
                    pred_cpu,
                    target_cpu,
                    num_classes,
                )
            ]
        )
    )


def specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 1,
    smooth: float = 1e-6,
) -> float:
    """Measure what fraction of true background was correctly rejected.

    Specificity exists because a model can capture foreground while also
    over-segmenting large amounts of healthy tissue. Clinically, low specificity
    means the model marks healthy tissue as diseased, which can lead to
    unnecessary follow-up, intervention, or unsafe planning margins. Specificity
    ranges from 0 to 1, where 1 means there are no false positives. It can
    mislead as a standalone metric because medical images are often mostly
    background, making true negatives large and specificity high by default.

    Formula:
    ``Specificity = TN / (TN + FP + smooth)``.
    Specificity is asymmetric with sensitivity. You cannot optimize both
    simultaneously without changing the decision threshold, so threshold policy
    is part of evaluation design rather than architecture design.

    Args:
        pred: Boolean or integer tensor of predicted labels shaped ``(H, W)`` or
            ``(D, H, W)`` for one sample.
        target: Boolean or integer tensor of ground-truth labels with the same
            shape as ``pred``.
        num_classes: Use ``1`` for binary masks or ``K`` for multiclass labels
            ``0`` through ``K - 1``. Multiclass specificity averages foreground
            one-vs-rest scores.
        smooth: Additive smoothing constant for stable denominators.

    Returns:
        Mean foreground specificity as a Python ``float``.
    """

    pred_cpu, target_cpu = _validate_metric_inputs(
        pred,
        target,
        num_classes=num_classes,
        smooth=smooth,
    )
    return float(
        np.mean(
            [
                _specificity_for_masks(pred_mask, target_mask, smooth)
                for pred_mask, target_mask in _foreground_masks(
                    pred_cpu,
                    target_cpu,
                    num_classes,
                )
            ]
        )
    )


def hausdorff_95(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Measure robust boundary error between two binary mask surfaces.

    HD95 measures the 95th percentile of bidirectional surface distances. It
    exists because overlap metrics can look acceptable even when a boundary is
    displaced in a clinically important region. Clinically, a HD95 of 2 mm on a
    liver segmentation means that 95% of the predicted and target surfaces are
    within 2 mm of each other. HD95 can mislead when either mask is empty,
    because surface distance is undefined, and disconnected prediction fragments
    can still affect the upper tail of distances.

    A mask surface is the set of foreground pixels or voxels adjacent to
    background. The 95th percentile is used instead of maximum Hausdorff
    distance because the maximum is dominated by a single outlier voxel or
    disconnected fragment, while HD95 remains robust enough to describe boundary
    quality. ``spacing`` matters because distances without physical spacing are
    in pixels, not millimetres, and are not comparable across datasets with
    different resolution. Prefer HD95 over Dice for boundary-sensitive tasks
    such as radiation treatment planning, where boundary errors translate
    directly into dose errors in healthy tissue.

    Args:
        pred: Boolean or binary tensor of predicted foreground shaped ``(H, W)``
            or ``(D, H, W)``.
        target: Boolean or binary tensor of target foreground with the same
            shape as ``pred``.
        spacing: Physical voxel spacing per axis, such as ``(1.0, 1.0)`` for
            isotropic 2D pixels or ``(3.0, 1.0, 1.0)`` for anisotropic 3D MRI.
            When omitted, distances are reported in pixel or voxel units.

    Returns:
        HD95 in spacing units as a Python ``float``.

    Raises:
        ValueError: If either mask is empty or any input contract is invalid.
    """

    pred_cpu, target_cpu = _validate_mask_inputs(pred, target)
    _validate_spacing(spacing, ndim=pred_cpu.ndim)

    pred_mask = pred_cpu.bool().numpy()
    target_mask = target_cpu.bool().numpy()

    if not pred_mask.any():
        raise ValueError("pred must contain at least one foreground value for hausdorff_95")
    if not target_mask.any():
        raise ValueError("target must contain at least one foreground value for hausdorff_95")

    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    pred_surface = pred_mask & ~binary_erosion(pred_mask)
    target_surface = target_mask & ~binary_erosion(target_mask)

    pred_coords = np.argwhere(pred_surface).astype(np.float64)
    target_coords = np.argwhere(target_surface).astype(np.float64)

    if spacing is not None:
        spacing_array = np.asarray(spacing, dtype=np.float64)
        pred_coords *= spacing_array
        target_coords *= spacing_array

    target_tree = cKDTree(target_coords)
    pred_tree = cKDTree(pred_coords)
    dist_pred_to_target, _ = target_tree.query(pred_coords)
    dist_target_to_pred, _ = pred_tree.query(target_coords)
    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])

    return float(np.percentile(all_distances, 95))


def evaluate(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 1,
    smooth: float = 1e-6,
    include_hausdorff: bool = False,
    spacing: tuple[float, ...] | None = None,
) -> SegmentationMetrics:
    """Compute the standard segmentation metric set for foreground classes.

    ``evaluate`` measures overlap, confusion-count behavior, and optionally
    boundary distance in one structured result. It exists to keep reports
    internally consistent: using the same class masks and smoothing convention
    across metrics avoids accidental differences between tables. Clinically, the
    aggregate fields summarize typical foreground behavior while ``per_class``
    shows which structure drove a score. The summary can mislead when a rare or
    clinically important class is hidden by macro-averaging, so inspect
    ``per_class`` before drawing conclusions.

    HD95 is opt-in because surface-distance computation is slower than overlap
    metrics and requires SciPy. ``per_class`` contains one ``ClassMetrics`` entry
    per foreground class, ordered by class index: length ``1`` for binary
    segmentation and length ``K - 1`` for ``K``-class multiclass segmentation.
    Background label ``0`` is excluded from ``per_class``.

    Args:
        pred: Boolean or integer tensor of predicted labels shaped ``(H, W)`` or
            ``(D, H, W)`` for one sample.
        target: Boolean or integer tensor of ground-truth labels with the same
            shape as ``pred``.
        num_classes: Use ``1`` for binary masks or ``K`` for multiclass labels
            ``0`` through ``K - 1``.
        smooth: Additive smoothing constant for overlap and confusion-count
            metrics.
        include_hausdorff: When ``True``, compute HD95 for each foreground class
            that is present in at least one mask.
        spacing: Optional physical spacing passed to ``hausdorff_95``.

    Returns:
        A ``SegmentationMetrics`` instance containing aggregate and per-class
        values.
    """

    pred_cpu, target_cpu = _validate_metric_inputs(
        pred,
        target,
        num_classes=num_classes,
        smooth=smooth,
    )
    if include_hausdorff:
        _validate_spacing(spacing, ndim=pred_cpu.ndim)

    per_class: list[ClassMetrics] = []
    hausdorff_values: list[float] = []
    for pred_mask, target_mask in _foreground_masks(pred_cpu, target_cpu, num_classes):
        class_hausdorff: float | None = None
        if include_hausdorff and (pred_mask.any() or target_mask.any()):
            if pred_mask.any() and target_mask.any():
                class_hausdorff = hausdorff_95(
                    pred_mask,
                    target_mask,
                    spacing=spacing,
                )
                hausdorff_values.append(class_hausdorff)

        per_class.append(
            ClassMetrics(
                dice=_dice_for_masks(pred_mask, target_mask, smooth),
                iou=_iou_for_masks(pred_mask, target_mask, smooth),
                sensitivity=_sensitivity_for_masks(pred_mask, target_mask, smooth),
                specificity=_specificity_for_masks(pred_mask, target_mask, smooth),
                hausdorff_95=class_hausdorff,
            )
        )

    return SegmentationMetrics(
        mean_dice=float(np.mean([class_metrics.dice for class_metrics in per_class])),
        mean_iou=float(np.mean([class_metrics.iou for class_metrics in per_class])),
        mean_sensitivity=float(
            np.mean([class_metrics.sensitivity for class_metrics in per_class])
        ),
        mean_specificity=float(
            np.mean([class_metrics.specificity for class_metrics in per_class])
        ),
        mean_hausdorff_95=(
            float(np.mean(hausdorff_values))
            if include_hausdorff and hausdorff_values
            else None
        ),
        per_class=tuple(per_class),
    )


def _validate_metric_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    smooth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_num_classes(num_classes)
    _validate_smooth(smooth)
    pred_cpu, target_cpu = _validate_mask_inputs(pred, target)
    _validate_label_values(pred_cpu, name="pred", num_classes=num_classes)
    _validate_label_values(target_cpu, name="target", num_classes=num_classes)
    return pred_cpu, target_cpu


def _validate_mask_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape; "
            f"got pred {tuple(pred.shape)} and target {tuple(target.shape)}"
        )
    if pred.ndim < 2:
        raise ValueError("pred and target must be at least 2-dimensional")
    return pred.detach().cpu(), target.detach().cpu()


def _validate_num_classes(num_classes: int) -> None:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise ValueError("num_classes must be a positive integer")
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")


def _validate_smooth(smooth: float) -> None:
    if smooth < 0:
        raise ValueError("smooth must be non-negative")


def _validate_spacing(spacing: tuple[float, ...] | None, *, ndim: int) -> None:
    if spacing is None:
        return
    if len(spacing) != ndim:
        raise ValueError(
            f"spacing must have length {ndim} to match pred and target dimensions; "
            f"got {len(spacing)}"
        )
    if any(axis_spacing <= 0 for axis_spacing in spacing):
        raise ValueError("spacing values must be positive")


def _validate_label_values(
    tensor: torch.Tensor,
    *,
    name: str,
    num_classes: int,
) -> None:
    if tensor.numel() == 0:
        return
    if tensor.is_floating_point():
        rounded = tensor.round()
        if not torch.equal(tensor, rounded):
            raise ValueError(
                f"{name} must contain binary mask values or integer class labels, not raw logits"
            )
        tensor = rounded
    if num_classes == 1:
        valid = (tensor == 0) | (tensor == 1)
        if not bool(valid.all()):
            raise ValueError(f"{name} must contain only 0/1 values when num_classes is 1")
        return
    valid = (tensor >= 0) & (tensor < num_classes)
    if not bool(valid.all()):
        raise ValueError(
            f"{name} must contain integer class labels from 0 to {num_classes - 1}"
        )


def _foreground_masks(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if num_classes == 1:
        return ((pred.bool(), target.bool()),)
    return tuple(
        (pred == class_index, target == class_index)
        for class_index in range(1, num_classes)
    )


def _dice_for_masks(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float,
) -> float:
    intersection = (pred_mask & target_mask).sum().item()
    pred_area = pred_mask.sum().item()
    target_area = target_mask.sum().item()
    return float((2 * intersection + smooth) / (pred_area + target_area + smooth))


def _iou_for_masks(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float,
) -> float:
    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    return float((intersection + smooth) / (union + smooth))


def _sensitivity_for_masks(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float,
) -> float:
    true_positive = (pred_mask & target_mask).sum().item()
    false_negative = (~pred_mask & target_mask).sum().item()
    return float(true_positive / (true_positive + false_negative + smooth))


def _specificity_for_masks(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float,
) -> float:
    true_negative = (~pred_mask & ~target_mask).sum().item()
    false_positive = (pred_mask & ~target_mask).sum().item()
    return float(true_negative / (true_negative + false_positive + smooth))


__all__ = [
    "ClassMetrics",
    "SegmentationMetrics",
    "dice_score",
    "evaluate",
    "hausdorff_95",
    "iou_score",
    "sensitivity",
    "specificity",
]
