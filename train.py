"""Training entrypoint for the Edge Impulse PyTorch FOMO learning block.

Edge Impulse passes the following arguments automatically:

    --info-file <file>        Path to train_input.json (training configuration).
    --data-directory <dir>    Directory with the NumPy train/test splits.
    --out-directory <dir>     Directory to write model artifacts to.

Custom parameters defined in parameters.json are passed as extra arguments:

    --epochs <int>
    --learning-rate <float>
    --batch-size <int>
    --object-weight <float>
    --export-pte <flag>

Input data (already pre-processed and pixel-scaled by Edge Impulse):

    X_split_train.npy / X_split_test.npy   float32, NHWC (batch, height, width, channels)
    Y_split_train.npy / Y_split_test.npy   JSON *text* (despite the .npy extension):
        [{"sampleId": <int>, "boundingBoxes": [{"label", "x", "y", "w", "h"}, ...]}, ...]

FOMO turns bounding boxes into a coarse class grid (the input downsampled by 8)
where each cell holds the class of the object centred on it, or background. We
train with a weighted cross-entropy over that grid and export an ONNX model that
outputs per-cell class probabilities (the FOMO output layer).
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import FomoExport, build_model

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch FOMO learning block")
    parser.add_argument("--info-file", type=str, required=False)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--out-directory", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--object-weight", type=float, default=100.0)
    parser.add_argument("--export-pte", action="store_true", default=False)
    # Ignore any unknown args Edge Impulse may add in the future.
    args, _ = parser.parse_known_args()
    return args


def load_split(data_dir: str, split: str) -> tuple[np.ndarray, list]:
    x = np.load(os.path.join(data_dir, f"X_split_{split}.npy"))
    # For object detection, Y_split_*.npy is a JSON *text* file (an array of
    # per-sample dicts), not a NumPy array – despite the .npy extension. Read it
    # as JSON, matching how Edge Impulse's own object-detection blocks load it.
    with open(os.path.join(data_dir, f"Y_split_{split}.npy"), "r") as f:
        labels = json.loads(f.read())
    return x, labels


def to_nchw(x: np.ndarray) -> np.ndarray:
    """Convert NHWC image data to contiguous NCHW for PyTorch.

    We force a C-contiguous copy so torch.export captures dim_order
    (0, 1, 2, 3); the XNNPACK backend rejects channels-last inputs during .pte
    lowering.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D NHWC image data, got shape {x.shape}. "
            "This block operates on object-detection image data."
        )
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))


def iter_boxes(sample: dict) -> list[dict]:
    if not isinstance(sample, dict):
        return []
    return sample.get("boundingBoxes") or []


def build_label_map(*label_sets: list) -> dict[int, int]:
    """Map raw bounding-box labels to contiguous foreground channels 1..K.

    Channel 0 is reserved for background. This is robust whether Edge Impulse
    emits 0- or 1-indexed labels.
    """
    raw = set()
    for labels in label_sets:
        for sample in labels:
            for box in iter_boxes(sample):
                raw.add(int(box["label"]))
    return {label: idx + 1 for idx, label in enumerate(sorted(raw))}


def resolve_num_classes(info_file: str | None, label_map: dict[int, int]) -> int:
    """Return the number of output channels (foreground classes + background).

    Edge Impulse declares the authoritative class list in train_input.json
    (passed via --info-file). Sizing the head from the data is unreliable: a
    split may not contain every class, producing a head whose width disagrees
    with Edge Impulse's evaluator and making metric calculation fail (reported
    as 0% accuracy). We size from the declared class list (+1 for background),
    and only fall back to the observed labels for local runs without an info
    file.
    """
    if info_file:
        try:
            with open(info_file) as f:
                classes = json.load(f).get("classes")
            if classes:
                return len(classes) + 1
        except (OSError, ValueError):
            pass
    return len(label_map) + 1


def boxes_to_grid(
    labels: list,
    label_map: dict[int, int],
    grid_h: int,
    grid_w: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Convert per-sample bounding boxes to a (N, grid_h, grid_w) class grid."""
    targets = np.zeros((len(labels), grid_h, grid_w), dtype=np.int64)
    for i, sample in enumerate(labels):
        for box in iter_boxes(sample):
            cx = box["x"] + box["w"] / 2.0
            cy = box["y"] + box["h"] / 2.0
            gx = min(int(cx / img_w * grid_w), grid_w - 1)
            gy = min(int(cy / img_h * grid_h), grid_h - 1)
            targets[i, gy, gx] = label_map[int(box["label"])]
    return targets


def make_loader(
    x: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    tensor_x = torch.from_numpy(x.astype(np.float32))
    tensor_y = torch.from_numpy(targets.astype(np.int64))
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Accuracy over foreground (object) cells only.

    Background dominates the grid, so plain accuracy is misleading; we report how
    often the model gets the class right on cells that actually contain an object.
    """
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        fg = targets > 0
        total += int(fg.sum().item())
        correct += int((preds[fg] == targets[fg]).sum().item())
    return correct / total if total else float("nan")


def export_onnx(model: nn.Module, sample_input: torch.Tensor, out_path: str) -> None:
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Saved ONNX model to {out_path}")


def export_pte(model: nn.Module, sample_input: torch.Tensor, out_path: str) -> None:
    """Export an ExecuTorch program (.pte) lowered to the XNNPACK backend.

    ExecuTorch is optional at runtime; if it isn't installed we warn and skip
    rather than failing the whole training job.
    """
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower
    except ImportError:
        print(
            "WARNING: executorch is not installed; skipping .pte export. "
            "Add 'executorch' to requirements.txt to enable it."
        )
        return

    model.eval()
    # XNNPACK only accepts contiguous (channels-first) inputs during lowering.
    sample_input = sample_input.contiguous(memory_format=torch.contiguous_format)
    try:
        exported = torch.export.export(model, (sample_input,))
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    except Exception as exc:  # noqa: BLE001 - .pte is an optional artifact
        print(
            "WARNING: ExecuTorch lowering failed; skipping .pte export "
            f"(the ONNX model is unaffected). Reason: {exc}"
        )
        return
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"Saved ExecuTorch program to {out_path}")


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, y_train = load_split(args.data_directory, "train")
    x_test, y_test = load_split(args.data_directory, "test")

    x_train = to_nchw(x_train)
    x_test = to_nchw(x_test) if x_test.size else x_test.reshape(0, *x_train.shape[1:])

    in_channels, img_h, img_w = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    label_map = build_label_map(y_train, y_test)
    if not label_map:
        raise ValueError(
            "No bounding boxes found in the training labels. This block requires "
            "an object-detection dataset labelled with bounding boxes."
        )
    num_classes = resolve_num_classes(args.info_file, label_map)  # incl. background

    model = build_model(in_channels=in_channels, num_classes=num_classes).to(device)

    # Derive the FOMO grid size from an actual forward pass so it always matches
    # the backbone's downsampling (8x) for the given input resolution.
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, img_h, img_w, device=device)
        grid_h, grid_w = model(dummy).shape[-2:]
    print(
        f"Channels: {in_channels}, classes (incl. background): {num_classes}, "
        f"input: {img_h}x{img_w}, grid: {grid_h}x{grid_w}, "
        f"train: {len(x_train)}, test: {len(x_test)}"
    )

    t_train = boxes_to_grid(y_train, label_map, grid_h, grid_w, img_h, img_w)
    t_test = (
        boxes_to_grid(y_test, label_map, grid_h, grid_w, img_h, img_w)
        if len(x_test)
        else np.zeros((0, grid_h, grid_w), dtype=np.int64)
    )

    train_loader = make_loader(x_train, t_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, t_test, args.batch_size, shuffle=False)

    # Weighted loss: background weight 1, objects weighted up so the model isn't
    # overwhelmed by the (dominant) background cells.
    weights = torch.full((num_classes,), args.object_weight, device=device)
    weights[0] = 1.0
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - "
            f"val_object_acc: {acc:.4f}"
        )

    os.makedirs(args.out_directory, exist_ok=True)
    model = model.to("cpu")
    export_model = FomoExport(model)
    # ascontiguousarray guarantees dim_order (0, 1, 2, 3) for the XNNPACK export.
    sample_input = torch.from_numpy(
        np.ascontiguousarray(x_train[:1], dtype=np.float32)
    )

    export_onnx(
        export_model, sample_input, os.path.join(args.out_directory, "model.onnx")
    )
    if args.export_pte:
        export_pte(
            export_model, sample_input, os.path.join(args.out_directory, "model.pte")
        )


if __name__ == "__main__":
    main()
