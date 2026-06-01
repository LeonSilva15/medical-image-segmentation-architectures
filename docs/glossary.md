# Glossary

This glossary collects recurring terms used across the book. Architecture pages
still explain terms in context, but this page gives readers one place to check
definitions.

| Term | Meaning |
| --- | --- |
| Semantic segmentation | Segmentation where each pixel or voxel receives a class label, without separating different objects of the same class. |
| Instance segmentation | Segmentation where separate objects receive separate IDs, not just foreground/background labels. |
| Promptable segmentation | Segmentation where user guidance such as points, boxes, masks, text, or memory controls which mask the model returns. |
| Foundation model | A broadly pretrained model adapted to downstream segmentation workflows; in this book, the term does not imply clinical readiness. |
| Volumetric segmentation | Segmentation over 3D images or patches, usually represented as `(B, C, D, H, W)`, with through-plane context. |
| Dense prediction | A task where the model produces a prediction at each pixel, voxel, or spatial location instead of one label for the whole image. |
| Tensor shape notation | The project convention for naming tensor dimensions, such as `(B, C, H, W)` for 2D batches and `(B, C, D, H, W)` for 3D batches. |
| Logits | Raw model scores before `sigmoid` or `softmax`. Use logits directly with logits-aware losses. |
| Probability map | A logit map converted to values interpretable as probabilities. |
| Feature map | An internal tensor of learned channels at a particular spatial resolution. |
| Voxel | A volume element in 3D data, analogous to a pixel in 2D data. |
| Encoder-decoder | A model pattern where an encoder compresses spatial features and a decoder restores dense spatial predictions. |
| Downsampling / upsampling | Downsampling reduces spatial resolution to increase context; upsampling restores resolution for dense output. |
| Skip connection | A path that carries earlier features into a later layer, often to restore spatial detail. |
| Bottleneck | The deepest, lowest-resolution representation in an encoder-decoder model. |
| Output stride | The ratio between input resolution and an internal feature-map resolution. |
| ASPP | Atrous Spatial Pyramid Pooling, a multi-scale context module using dilated convolutions. |
| Atrous / dilated convolution | A convolution that spaces out kernel samples to increase receptive field without adding pooling. |
| Residual block / shortcut | A block that adds a shortcut path around layers so the block can learn a correction to its input. |
| Attention gate | A module that uses learned weights to filter or emphasize features, often before merging skip features into a decoder. |
| Deep supervision | Extra training losses attached to intermediate decoder outputs to guide earlier layers during optimization. |
| Tokenization and patch embedding | The process of turning image or volume patches into token features consumed by Transformer-style layers. |
| Shifted-window attention | Swin-style local attention where window boundaries shift between blocks so neighboring windows can exchange context. |
| Prompt | User-provided guidance such as a point, box, mask, text label, or memory state. |
| Prompt-conditioned mask decoder | A decoder that combines image features with prompt features to produce the requested mask. |
| Memory-conditioned prompting | Prompting that uses stored context from prior images, slices, frames, or interactions to influence the next mask. |
| Patch-based training | Training on cropped image or volume regions instead of full scans to control memory use and sampling. |
| Sliding-window inference | Applying a model to overlapping patches and stitching the patch predictions into a full-size output. |
| Voxel spacing / resampling | Voxel spacing is the physical size represented by each voxel; resampling changes an image or mask to a chosen spacing or grid. |
| Dataset fingerprint | A dataset summary, such as shapes, spacing, modalities, labels, and intensity behavior, used to plan preprocessing or model settings. |
| Dataset leakage | Evaluation contamination caused by related patients, cases, slices, patches, preprocessing statistics, or duplicates crossing data splits. |
| External validation | Evaluation on data from a separate site, scanner, institution, acquisition protocol, or domain beyond internal splits. |
| Threshold policy | The chosen rule for converting probabilities into binary predictions; changing it changes precision and sensitivity tradeoffs. |
| Dice score | An overlap metric: twice the intersection divided by predicted plus target area. |
| IoU | Intersection over Union, also called Jaccard. |
| HD95 | The 95th percentile Hausdorff distance, a boundary-distance metric less sensitive to a single worst outlier than full Hausdorff distance. |
| Surface Dice | A boundary agreement metric that counts surfaces as matching when they are within a chosen distance tolerance. |
| Average precision | An instance-segmentation metric that summarizes precision and recall over one or more IoU thresholds. |
| Sparse annotation / loss mask | Training with only some labeled pixels or voxels, where unlabeled locations must be ignored or handled explicitly in the loss. |
| PHI / DICOM header | Protected health information and medical-image metadata that may contain patient identifiers or clinical details. |
| Clinical-readiness claim | A claim that a model is safe, approved, or ready for diagnosis or deployment; this project treats such claims as out of scope. |
| Star-convex polyhedron | A 3D StarDist-style object shape represented by distances from an object center along fixed rays. |
| Radial distance map | A map of center-to-boundary distances used to parameterize star-convex object proposals. |
| Non-maximum suppression | A postprocessing step that removes overlapping duplicate object proposals while keeping the strongest candidates. |
| Flow field | A dense vector output that indicates how pixels or voxels should move or group during instance reconstruction. |
| Gradient tracking | Following a predicted vector or gradient field from pixels or voxels toward object centers or regions. |
| Normalized cut | A graph-partitioning objective that groups similar neighboring pixels or voxels into coherent regions while penalizing weak separation. |
