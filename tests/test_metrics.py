import pytest
import torch

from medseg_architectures import (
    ClassMetrics,
    SegmentationMetrics,
    dice_score,
    evaluate,
    hausdorff_95,
    iou_score,
    sensitivity,
    specificity,
)


def test_dice_score_is_one_for_identical_masks() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    assert dice_score(target, target) == pytest.approx(1.0)


def test_dice_score_is_zero_for_disjoint_masks() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[2:8, 2:8] = True
    target[20:26, 20:26] = True

    assert dice_score(pred, target, smooth=0.0) == 0.0


def test_dice_score_is_between_zero_and_one_for_partial_overlap() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[8:24, 8:24] = True
    target[16:30, 16:30] = True

    score = dice_score(pred, target)

    assert 0.0 < score < 1.0


def test_dice_score_handles_empty_pred_and_empty_target() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)

    assert dice_score(pred, target) == pytest.approx(1.0)


def test_dice_score_handles_empty_pred_nonempty_target() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    assert dice_score(pred, target) == pytest.approx(0.0, abs=1e-5)


def test_dice_score_multiclass_averages_foreground_classes() -> None:
    pred = torch.zeros(4, 4, dtype=torch.long)
    target = torch.zeros(4, 4, dtype=torch.long)
    pred[0:2, 0:2] = 1
    target[0:2, 0:2] = 1
    pred[2:4, 0:2] = 2
    target[2:4, 2:4] = 2

    assert dice_score(pred, target, num_classes=3, smooth=0.0) == pytest.approx(0.5)


def test_iou_score_is_one_for_identical_masks() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    assert iou_score(target, target) == pytest.approx(1.0)


def test_iou_score_is_zero_for_disjoint_masks() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[2:8, 2:8] = True
    target[20:26, 20:26] = True

    assert iou_score(pred, target, smooth=0.0) == 0.0


def test_iou_score_is_less_than_or_equal_to_dice_for_same_inputs() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[8:24, 8:24] = True
    target[12:28, 12:28] = True

    assert iou_score(pred, target) <= dice_score(pred, target)


def test_sensitivity_is_one_when_all_foreground_is_captured() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True
    pred[6:26, 6:26] = True

    assert sensitivity(pred, target, smooth=0.0) == 1.0


def test_sensitivity_is_zero_when_no_foreground_is_captured() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    assert sensitivity(pred, target, smooth=0.0) == 0.0


def test_specificity_is_one_when_no_false_positives() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[10:20, 10:20] = True
    target[8:24, 8:24] = True

    assert specificity(pred, target, smooth=0.0) == 1.0


def test_sensitivity_plus_miss_rate_is_one() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[8:16, 8:24] = True
    target[8:24, 8:24] = True
    true_positive = (pred & target).sum().item()
    false_negative = (~pred & target).sum().item()
    miss_rate = false_negative / (true_positive + false_negative)

    assert sensitivity(pred, target, smooth=0.0) + miss_rate == pytest.approx(1.0)


def test_hausdorff_95_is_zero_for_identical_masks() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    assert hausdorff_95(target, target) == pytest.approx(0.0)


def test_hausdorff_95_increases_with_boundary_offset() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True
    pred[8:24, 11:27] = True

    assert hausdorff_95(pred, target) == pytest.approx(3.0)


def test_hausdorff_95_respects_spacing() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True
    pred[10:26, 8:24] = True

    assert hausdorff_95(pred, target, spacing=(2.0, 1.0)) == pytest.approx(4.0)


def test_hausdorff_95_raises_for_empty_pred() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    with pytest.raises(ValueError, match="pred"):
        hausdorff_95(pred, target)


def test_hausdorff_95_raises_for_empty_target() -> None:
    pred = torch.zeros(32, 32, dtype=torch.bool)
    target = torch.zeros(32, 32, dtype=torch.bool)
    pred[8:24, 8:24] = True

    with pytest.raises(ValueError, match="target"):
        hausdorff_95(pred, target)


def test_evaluate_returns_all_fields_for_identical_masks() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    result = evaluate(target, target)

    assert isinstance(result, SegmentationMetrics)
    assert isinstance(result.per_class[0], ClassMetrics)
    assert result.mean_dice == pytest.approx(1.0)
    assert result.mean_iou == pytest.approx(1.0)
    assert result.mean_sensitivity == pytest.approx(1.0, abs=1e-5)
    assert result.mean_specificity == pytest.approx(1.0, abs=1e-5)


def test_evaluate_per_class_length_equals_num_foreground_classes() -> None:
    pred = torch.zeros(16, 16, dtype=torch.long)
    target = torch.zeros(16, 16, dtype=torch.long)
    pred[2:6, 2:6] = 1
    target[2:6, 2:6] = 1
    pred[8:12, 8:12] = 2
    target[8:12, 8:12] = 2

    result = evaluate(pred, target, num_classes=3)

    assert len(result.per_class) == 2


def test_evaluate_hausdorff_is_none_when_not_requested() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    result = evaluate(target, target)

    assert result.mean_hausdorff_95 is None
    assert result.per_class[0].hausdorff_95 is None


def test_evaluate_hausdorff_is_not_none_when_requested() -> None:
    target = torch.zeros(32, 32, dtype=torch.bool)
    target[8:24, 8:24] = True

    result = evaluate(target, target, include_hausdorff=True)

    assert result.mean_hausdorff_95 == pytest.approx(0.0)
    assert result.per_class[0].hausdorff_95 == pytest.approx(0.0)


def test_metrics_raise_for_mismatched_shapes() -> None:
    pred = torch.zeros(16, 16, dtype=torch.bool)
    target = torch.zeros(16, 15, dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape"):
        dice_score(pred, target)


def test_metrics_raise_for_negative_smooth() -> None:
    pred = torch.zeros(16, 16, dtype=torch.bool)
    target = torch.zeros(16, 16, dtype=torch.bool)

    with pytest.raises(ValueError, match="smooth"):
        iou_score(pred, target, smooth=-1.0)


def test_metrics_raise_for_zero_num_classes() -> None:
    pred = torch.zeros(16, 16, dtype=torch.bool)
    target = torch.zeros(16, 16, dtype=torch.bool)

    with pytest.raises(ValueError, match="num_classes"):
        sensitivity(pred, target, num_classes=0)


def test_hausdorff_95_raises_for_wrong_spacing_length() -> None:
    pred = torch.zeros(16, 16, dtype=torch.bool)
    target = torch.zeros(16, 16, dtype=torch.bool)

    with pytest.raises(ValueError, match="spacing"):
        hausdorff_95(pred, target, spacing=(1.0, 1.0, 1.0))
