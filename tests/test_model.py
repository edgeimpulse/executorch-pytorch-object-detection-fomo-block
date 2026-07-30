"""Smoke tests for the FOMO model and data handling.

Run with: python -m pytest
"""

import numpy as np
import torch

from model import FomoExport, build_model
from train import boxes_to_grid, build_label_map, to_nchw


def test_forward_downsamples_by_8():
    model = build_model(in_channels=3, num_classes=3)
    out = model(torch.randn(2, 3, 96, 96))
    # 96 / 8 = 12 grid, 3 classes (incl. background).
    assert out.shape == (2, 3, 12, 12)


def test_export_softmax_normalises_channels():
    model = build_model(in_channels=1, num_classes=2)
    export = FomoExport(model)
    out = export(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 2, 8, 8)
    # Softmax over the class dimension sums to 1 per cell.
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_to_nchw():
    nhwc = np.zeros((4, 96, 96, 3), dtype=np.float32)
    assert to_nchw(nhwc).shape == (4, 3, 96, 96)


def test_build_label_map_is_contiguous():
    labels = [
        {"sampleId": 1, "boundingBoxes": [{"label": 5, "x": 0, "y": 0, "w": 2, "h": 2}]},
        {"sampleId": 2, "boundingBoxes": [{"label": 9, "x": 1, "y": 1, "w": 2, "h": 2}]},
    ]
    label_map = build_label_map(labels)
    # Foreground channels start at 1 (0 is background).
    assert label_map == {5: 1, 9: 2}


def test_boxes_to_grid_places_centroid():
    labels = [
        {
            "sampleId": 1,
            "boundingBoxes": [{"label": 5, "x": 40, "y": 40, "w": 16, "h": 16}],
        }
    ]
    label_map = build_label_map(labels)
    grid = boxes_to_grid(labels, label_map, grid_h=12, grid_w=12, img_h=96, img_w=96)
    assert grid.shape == (1, 12, 12)
    # Centroid at (48, 48) -> cell (6, 6), class channel 1.
    assert grid[0, 6, 6] == 1
    # Everything else is background.
    assert grid.sum() == 1
