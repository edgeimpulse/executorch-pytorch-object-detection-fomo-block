# PyTorch FOMO object-detection block (ExecuTorch)

An [Edge Impulse custom learning block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks) that trains a compact, fully-convolutional [FOMO](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo)-style object detector in PyTorch and exports:

- `model.onnx` — consumed by Edge Impulse (FOMO output layer, auto-converted to TFLite for deployment).
- `model.pte` — an optional [ExecuTorch](https://pytorch.org/executorch/) program for native PyTorch on-device inference.

This is the object-detection variant of the ExecuTorch end-to-end series (see also the image and time-series blocks).

## Files

| File | Purpose |
|------|---------|
| `parameters.json` | Block metadata (`operatesOn: object_detection`, `objectDetectionLastLayer: fomo`) and the Studio parameters. |
| `model.py` | `FomoNet` (backbone + 1x1 head) and `FomoExport` (adds the softmax used at inference). |
| `train.py` | Training entrypoint. Converts boxes to a centroid grid, trains, and exports ONNX (+ `.pte`). |
| `Dockerfile` | Builds the training container. |
| `requirements.txt` | Pinned Python dependencies. |
| `tests/test_model.py` | Smoke tests for the model, grid conversion, and data handling. |

## How it works

FOMO doesn't regress bounding boxes; it predicts a coarse class **grid** (the input downsampled by 8), where each cell holds the class of the object centred on it, or background. The block:

1. Loads `X_split_{train,test}.npy` (NHWC float32 images) and `Y_split_{train,test}.npy` (per-sample bounding-box dicts) from `--data-directory`.
2. Transposes NHWC → NCHW for PyTorch (Edge Impulse handles the reverse on-device because we output ONNX).
3. Builds a contiguous label map (foreground classes → channels `1..K`, background is channel `0`) and converts each box's centroid into a target grid cell.
4. Trains `FomoNet` with a **weighted** cross-entropy (background weight 1, objects weighted by `--object-weight`) so the dominant background cells don't overwhelm training.
5. Wraps the trained model with a channel-wise softmax and writes `model.onnx` (and `model.pte` when `--export-pte` is set) to `--out-directory`.

The grid size is derived from an actual forward pass, so it always matches the backbone's 8× downsampling for the given input resolution (e.g. 96×96 → 12×12).

## Test locally

Create an object-detection project (labeling method: **Bounding boxes**), add an **Image** processing block and this learning block, then:

```bash
# Pull the latest processed data into ./input
edge-impulse-blocks runner --download-data input/

# Build and run the training container
docker build -t pytorch-fomo-deploy .
docker run --rm -v "$PWD":/app pytorch-fomo-deploy \
    --data-directory /app/input \
    --out-directory /app/out \
    --epochs 60 --learning-rate 0.001 --batch-size 16 \
    --object-weight 100 --export-pte
```

After training you'll find `out/model.onnx` (and `out/model.pte`).

Run the unit tests without Docker:

```bash
pip install -r requirements.txt pytest
python -m pytest
```

## Push to Edge Impulse

```bash
edge-impulse-blocks init   # first time only
edge-impulse-blocks push
```

The block then appears under **Create impulse → Add learning block** (for object-detection projects) in Studio.

## Notes

- Use **square** input resolutions that are multiples of 8 (e.g. 96×96, 160×160). FOMO is fully convolutional, so only the aspect ratio is fixed.
- FOMO outputs **centroids**, not boxes — object size isn't recovered. It works best when objects are similar in size and not touching. See the [object detection with centroids](https://docs.edgeimpulse.com/tutorials/end-to-end/object-detection-centroids) tutorial.
- The `.pte` file is **not** used by the Edge Impulse SDK. It's included so you can carry the same trained weights into a native ExecuTorch runtime (see the `executorch-deploy` block).
- Pin `torch` and `executorch` to compatible versions. If ExecuTorch isn't installed, `.pte` export is skipped with a warning instead of failing the job.
