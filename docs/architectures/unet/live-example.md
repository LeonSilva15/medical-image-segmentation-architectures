# U-Net Live Example

This page is reserved for a future interactive or executable U-Net example.

## Planned Direction

The live example should run on synthetic data and should make the model behavior
easy to inspect without requiring clinical files or external datasets.

Good first targets:

- Generate a synthetic input tensor.
- Run `UNet2D` in evaluation mode.
- Display input shape, output shape, and parameter count.
- Optionally show a toy mask-like tensor once visualization support exists.

The example must remain educational only. It should not load private medical
images, PHI, patient identifiers, DICOM headers, or clinical data.

