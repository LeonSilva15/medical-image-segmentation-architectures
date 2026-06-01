# Evaluation Metrics

Segmentation metrics answer a question that pixel accuracy cannot: does the
model find the right structure? A model that predicts background everywhere can
exceed 99% pixel accuracy on a typical brain lesion dataset and still be useless.
The metrics on this page are designed to measure overlap and boundary quality in
a way that is meaningful for clinical use.

For the basic shape conventions and loss functions used during training, see
[Training And Evaluation Basics](training-and-evaluation-basics.md). For
implementations you can import directly, see the
`medseg_architectures.metrics` module.

## Metric Overview

| Metric | Range | Sensitive to | Typical use |
| --- | --- | --- | --- |
| Dice | 0 to 1, higher is better | Foreground overlap, class imbalance | Primary medical segmentation overlap metric |
| IoU / Jaccard | 0 to 1, higher is better | False positives and false negatives through the union | Computer vision benchmarks and stricter overlap reporting |
| Sensitivity | 0 to 1, higher is better | Missed foreground, false negatives | Screening tasks where missing disease is costly |
| Specificity | 0 to 1, higher is better | Over-segmentation, false positives | Ruling out healthy tissue and ROC-style analysis |
| HD95 | 0 to infinity, lower is better | Boundary displacement after excluding top 5% outliers | Boundary-sensitive planning and navigation tasks |

## Dice Coefficient

$$
\mathrm{Dice}(P, T) = \frac{2|P \cap T|}{|P| + |T|}
$$

Dice measures how much the predicted foreground region overlaps the target
foreground region. Geometrically, it takes the shared area twice and divides by
the total area counted from both masks.

The factor of 2 is not arbitrary. Suppose the prediction has 80 foreground
pixels, the target has 100 foreground pixels, and 70 pixels overlap. The
intersection alone, 70, is not normalized. Dividing by the average mask size
gives:

$$
\frac{70}{(80 + 100) / 2} = \frac{2 \times 70}{80 + 100} = 0.778
$$

When prediction and target are identical, the intersection equals both mask
sizes, so Dice becomes 1.0.

Dice is preferred over pixel accuracy because medical images are usually
background-heavy. In a 512 by 512 slice with a 2,000-pixel lesion, a model that
predicts background everywhere gets 260,144 of 262,144 pixels correct, or about
0.992 accuracy. Its Dice for the lesion is 0.0 because the predicted lesion area
is empty. Accuracy answers whether most pixels are easy background; Dice answers
whether the structure was found.

Empty masks need an explicit convention. If both prediction and target are
empty, this repository scores Dice as 1.0 for that class. That is the right
convention when evaluating absent structures: if a patient has no visible lesion
and the model predicts no lesion, the class-specific segmentation behavior is
correct. If the target is empty but the prediction is not, the false positive
still lowers the score.

For multiclass segmentation, compute Dice one class at a time using a one-vs-rest
mask:

$$
\mathrm{Dice}_k = \frac{2|\{x : P_x = k\} \cap \{x : T_x = k\}|}{|\{x : P_x = k\}| + |\{x : T_x = k\}|}
$$

Report the per-class scores before averaging. A high mean can hide a small but
clinically important class.

## IoU / Jaccard

$$
\mathrm{IoU}(P, T) = \frac{|P \cap T|}{|P \cup T|}
$$

Intersection over Union, also called Jaccard, divides the shared foreground by
the total region covered by either mask. It is stricter than Dice for the same
prediction because the union includes all false-positive and false-negative
foreground.

The relationship to Dice follows directly from the two definitions:

$$
\mathrm{IoU} =
\frac{|P \cap T|}{|P| + |T| - |P \cap T|}
=
\frac{\mathrm{Dice}}{2 - \mathrm{Dice}}
$$

The consequence is important: Dice and IoU always rank predictions identically.
Choosing between them is about convention and audience, not extra information.
Computer vision benchmarks such as COCO and Pascal VOC commonly use IoU. Most
medical segmentation papers and challenge reports use Dice as the primary
metric, so reviewers usually expect it.

## Sensitivity and Specificity

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}
\qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

Sensitivity is recall: the fraction of true foreground that the model captured.
Specificity is the fraction of true background that the model correctly rejected.
In ROC analysis, `1 - specificity` is the false positive rate.

Both metrics depend on the threshold used to turn probabilities into a binary
mask. If the threshold is lowered, more pixels become foreground: sensitivity
usually rises because fewer true positives are missed, while specificity usually
falls because more healthy tissue is flagged. Raising the threshold moves the
point in the opposite direction. The ROC curve is the trace of these tradeoffs
as the threshold changes.

Clinical interpretation is asymmetric. For a tumor screening model, missing a
real tumor, which appears as low sensitivity, is usually worse than flagging
healthy tissue, which appears as low specificity. For a surgical planning tool,
over-segmenting healthy tissue may cause equal harm because the mask can affect
where margins, instruments, or dose plans are placed. Neither metric alone is
sufficient.

Specificity also has a base-rate problem. Medical images contain many more
background pixels than foreground pixels, so true negatives are abundant. A model
can have high specificity by default while still missing the structure of
interest. Report sensitivity and specificity together with class prevalence so
readers can see the denominator behind each value.

## Hausdorff Distance at the 95th Percentile (HD95)

HD95 measures boundary error. The surface of a mask is the set of foreground
pixels or voxels touching background. For two overlapping rectangles, most
boundary points may lie directly on top of each other, while the shifted edge may
sit two pixels away from the target edge. Surface distance asks, for each
boundary point on one mask, how far away the nearest boundary point on the other
mask is, then repeats the question in the other direction.

The maximum Hausdorff distance uses the single worst boundary distance. That is
often too brittle. Imagine 100 boundary distances where 99 values are between 0
and 2 mm, but one disconnected prediction fragment is 40 mm away. The maximum is
40 mm, which describes the fragment more than the main segmentation. HD95
reports the 95th percentile instead, so it remains stable while still measuring
the upper tail of boundary error.

Physical spacing is essential. HD95 without spacing is measured in pixels or
voxels, not millimetres. Two CT scans can have the same array shape but different
slice thickness. A two-voxel through-plane boundary error may mean 2 mm in one
dataset and 10 mm in another. This is why HD95 should be reported in physical
units whenever voxel spacing is known, especially for anisotropic data such as
CT with thick axial slices.

HD95 matters when the clinical task is boundary-sensitive. Radiation treatment
planning uses boundaries to shape dose. Surgical navigation may move tools
toward boundaries. Tumor volume estimation changes when a boundary expands or
contracts. Dice can be sufficient for rough screening tasks,
classification-adjacent problems, or experiments where finding the approximate
structure is the clinical question and exact boundary placement is secondary.

## Choosing Metrics For Your Experiment

Always report Dice as the primary metric. It is the field standard for medical
segmentation, and readers expect it.

Add HD95 when the clinical task is boundary-sensitive. Define the spacing units
in the methods section so the number can be interpreted physically.

Report sensitivity and specificity when the asymmetry between missing true
positives and creating false positives is clinically meaningful. That includes
most clinical tasks, but the emphasis changes: screening usually prioritizes
sensitivity, while planning workflows often care strongly about both missed
foreground and over-segmentation.

## Code Examples

### Binary Segmentation Evaluation

```python
import torch
from medseg_architectures import UNet2D
from medseg_architectures.metrics import evaluate

torch.manual_seed(0)
model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
model.eval()

image = torch.randn(1, 1, 64, 64)
with torch.no_grad():
    logits = model(image)
pred = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0)  # (H, W) bool

# Synthetic ground truth: a centred square of foreground
target = torch.zeros(64, 64, dtype=torch.bool)
target[24:40, 24:40] = True

result = evaluate(pred, target, num_classes=1)
print(f"Dice:        {result.mean_dice:.4f}")
print(f"IoU:         {result.mean_iou:.4f}")
print(f"Sensitivity: {result.mean_sensitivity:.4f}")
print(f"Specificity: {result.mean_specificity:.4f}")
```

### HD95 With Physical Spacing

```python
import torch
from medseg_architectures import UNet2D
from medseg_architectures.metrics import evaluate

torch.manual_seed(0)
model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
model.eval()

image = torch.randn(1, 1, 64, 64)
with torch.no_grad():
    logits = model(image)

# Use a permissive threshold so the untrained synthetic model produces foreground.
pred = (torch.sigmoid(logits) > 0.4).squeeze(0).squeeze(0)

# Synthetic ground truth: a centred square of foreground
target = torch.zeros(64, 64, dtype=torch.bool)
target[24:40, 24:40] = True

result_with_hd95 = evaluate(
    pred,
    target,
    num_classes=1,
    include_hausdorff=True,
    spacing=(1.5, 1.5),  # 1.5mm isotropic in-plane spacing
)
print(f"HD95: {result_with_hd95.mean_hausdorff_95:.2f} mm")
```
