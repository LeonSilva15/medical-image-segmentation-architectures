from medseg_architectures import (
    UNet2D,
    count_parameters,
    shape_trace,
    summarize_parameters,
)


def test_count_parameters_matches_manual_total_for_unet_2d() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8))

    manual_total = sum(parameter.numel() for parameter in model.parameters())
    summary = summarize_parameters(model)

    assert count_parameters(model, trainable_only=False) == manual_total
    assert count_parameters(model) == manual_total
    assert summary.total == manual_total
    assert summary.trainable == manual_total
    assert summary.frozen == 0


def test_count_parameters_excludes_frozen_parameters_when_requested() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8))
    frozen_count = sum(parameter.numel() for parameter in model.output_conv.parameters())
    total = count_parameters(model, trainable_only=False)

    for parameter in model.output_conv.parameters():
        parameter.requires_grad = False

    summary = summarize_parameters(model)

    assert count_parameters(model, trainable_only=False) == total
    assert count_parameters(model) == total - frozen_count
    assert summary.total == total
    assert summary.trainable == total - frozen_count
    assert summary.frozen == frozen_count


def test_shape_trace_records_unet_2d_output_conv_shape() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8))

    trace = shape_trace(model, (1, 1, 32, 40))
    output_entry = next(entry for entry in trace if entry.module_name == "output_conv")

    assert output_entry.module_type == "Conv2d"
    assert output_entry.output_shape == (1, 2, 32, 40)


def test_shape_trace_supports_odd_unet_2d_input_size() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(8, 16, 32))

    trace = shape_trace(model, (1, 1, 65, 73))
    output_entry = next(entry for entry in trace if entry.module_name == "output_conv")

    assert output_entry.output_shape == (1, 2, 65, 73)


def test_shape_trace_restores_original_training_modes() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8))
    model.train()
    model.output_conv.eval()

    shape_trace(model, (1, 1, 32, 32))

    assert model.training
    assert model.down_blocks[0].training
    assert not model.output_conv.training
