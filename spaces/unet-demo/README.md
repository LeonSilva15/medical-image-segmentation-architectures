---
title: U-Net Segmentation Demo
emoji: 🧠
sdk: gradio
python_version: "3.11"
app_file: app.py
tags:
  - image-segmentation
  - medical-imaging
  - unet
  - education
---

# U-Net Segmentation Demo

Educational demo only. Not for clinical diagnosis. Do not upload private medical images, PHI, DICOM headers, or clinical data.

This Space runs a tiny U-Net on deterministic synthetic geometric examples. It
does not load patient data, clinical files, DICOM headers, model weights, or
external datasets. The model is trained in memory on synthetic shapes when the
app starts or first runs.
