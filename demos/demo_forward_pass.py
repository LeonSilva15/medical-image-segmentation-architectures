"""Run a synthetic tensor through the U-Net baseline."""

import torch

from medseg_architectures import UNet2D, shape_trace, summarize_parameters


def main() -> None:
    torch.manual_seed(7)
    model = UNet2D(in_channels=1, out_channels=2, features=(16, 32, 64))
    model.eval()

    x = torch.randn(1, 1, 65, 73)
    with torch.no_grad():
        y = model(x)

    parameter_summary = summarize_parameters(model)
    print(f"input_shape={tuple(x.shape)}")
    print(f"output_shape={tuple(y.shape)}")
    print(
        "parameters="
        f"total:{parameter_summary.total}, "
        f"trainable:{parameter_summary.trainable}, "
        f"frozen:{parameter_summary.frozen}"
    )

    print("shape_trace:")
    interesting_module_types = {"Conv2d", "ConvTranspose2d", "MaxPool2d"}
    for entry in shape_trace(model, tuple(x.shape)):
        if entry.module_type in interesting_module_types:
            print(
                f"  {entry.index:02d} {entry.module_name} "
                f"({entry.module_type}): {entry.input_shape} -> {entry.output_shape}"
            )


if __name__ == "__main__":
    main()
