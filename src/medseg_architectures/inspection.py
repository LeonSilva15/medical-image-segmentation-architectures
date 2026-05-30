"""Educational utilities for inspecting PyTorch segmentation models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

TensorShape: TypeAlias = tuple[int, ...]
ShapeValue: TypeAlias = TensorShape | tuple["ShapeValue", ...] | str


@dataclass(frozen=True)
class ParameterSummary:
    """Total, trainable, and frozen parameter counts for a model."""

    total: int
    trainable: int
    frozen: int


@dataclass(frozen=True)
class ShapeTraceEntry:
    """One leaf module's tensor shapes from a synthetic forward pass."""

    index: int
    module_name: str
    module_type: str
    input_shape: ShapeValue
    output_shape: ShapeValue


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally excluding frozen parameters.

    Args:
        model: PyTorch module to inspect.
        trainable_only: When ``True``, count only parameters with
            ``requires_grad=True``.

    Returns:
        Number of scalar parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def summarize_parameters(model: nn.Module) -> ParameterSummary:
    """Return total, trainable, and frozen parameter counts for ``model``."""

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return ParameterSummary(total=total, trainable=trainable, frozen=total - trainable)


def shape_trace(model: nn.Module, input_shape: Sequence[int]) -> list[ShapeTraceEntry]:
    """Trace leaf-module tensor shapes using a synthetic CPU input.

    The utility switches the model to evaluation mode during the trace, runs a
    single forward pass under ``torch.no_grad()``, and restores every module's
    previous training/evaluation flag afterward.

    Args:
        model: PyTorch module to inspect.
        input_shape: Shape for a synthetic tensor, such as ``(1, 1, 64, 64)``.

    Returns:
        Ordered entries for leaf modules that ran during the forward pass.
    """

    normalized_shape = _normalize_input_shape(input_shape)
    entries: list[ShapeTraceEntry] = []
    training_states = {module: module.training for module in model.modules()}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str, module_type: str):
        def hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            entries.append(
                ShapeTraceEntry(
                    index=len(entries),
                    module_name=module_name,
                    module_type=module_type,
                    input_shape=_shape_value_from_hook_inputs(inputs),
                    output_shape=_shape_value(output),
                )
            )

        return hook

    try:
        for name, module in model.named_modules():
            if name and not list(module.children()):
                hooks.append(
                    module.register_forward_hook(
                        make_hook(name, module.__class__.__name__)
                    )
                )

        sample = torch.zeros(normalized_shape, device=torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            model(sample)
    finally:
        for handle in hooks:
            handle.remove()
        for module, was_training in training_states.items():
            module.train(was_training)

    return entries


def _normalize_input_shape(input_shape: Sequence[int]) -> TensorShape:
    shape = tuple(int(dim) for dim in input_shape)
    if not shape:
        raise ValueError("input_shape must contain at least one dimension")
    if any(dim < 1 for dim in shape):
        raise ValueError("input_shape dimensions must be positive")
    return shape


def _shape_value_from_hook_inputs(inputs: tuple[object, ...]) -> ShapeValue:
    if len(inputs) == 1:
        return _shape_value(inputs[0])
    return tuple(_shape_value(input_value) for input_value in inputs)


def _shape_value(value: object) -> ShapeValue:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return tuple(_shape_value(item) for item in value)
    return type(value).__name__
