"""Run a synthetic tensor through the U-Net baseline."""

import torch

from medseg_architectures import UNet2D


def main() -> None:
    torch.manual_seed(7)
    model = UNet2D(in_channels=1, out_channels=2, features=(16, 32, 64))
    model.eval()

    x = torch.randn(1, 1, 65, 73)
    with torch.no_grad():
        y = model(x)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"input_shape={tuple(x.shape)}")
    print(f"output_shape={tuple(y.shape)}")
    print(f"parameter_count={parameter_count}")


if __name__ == "__main__":
    main()
