# cellpose-biosensor

Segmentation of individual fluorescent bacteria in Airyscan confocal images of
mouse colon tissue sections, with per-cell fluorescence intensity measurement.

## Biological setup

| Channel | Fluorophore | Purpose |
|---------|-------------|---------|
| C1 | BFP (constitutive) | Segmentation input + normalization denominator |
| C2 | GFP (osmotic-stress promoter) | Reporter — OFF in low osmolality |
| C3 | RFP (oxidative-stress promoter) | Reporter — measured as RFP/BFP |
| C4 | SYTOX far-red | Host nuclear stain — exclusion mask |

## Goal

Measure per-cell GFP/BFP and RFP/BFP ratios to quantify single-cell stress
reporter activity across samples. Segment bacteria using the BFP channel.

## Approach

Fine-tune Cellpose-SAM (Cellpose 4.x, default `cpsam` model) on manually
annotated tiles of the BFP channel from representative images (bacteria in
non-cleared colon tissue with fecal autofluorescence). Apply resulting model to
all 16-bit stacks. Extract intensity measurements from raw 16-bit data using
model-generated masks.

## Key challenges

- Fecal autofluorescence: broad-spectrum signal co-localising with bacteria
- Non-cleared tissue: host cell autofluorescence, out-of-plane blur
- Large stitched images (~6300×6300 px) — must be tiled for model input

## Environment

Requires macOS 14+ (Sonoma or later) for MPS (Apple Silicon GPU) support,
or Linux/Windows with CUDA for NVIDIA GPU.

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate cellpose-biosensor

# Install src package as editable (required for notebook imports)
pip install -e .

# Verify setup
python -c "
from cellpose import models
import torch
model = models.CellposeModel(gpu=True)
print('MPS:', torch.backends.mps.is_available())
print('model device:', next(model.net.parameters()).device)
"
```

## Software versions (as of initial setup)

- Python 3.10.20
- PyTorch 2.11.0
- Cellpose 4.1.1 (cpsam model — Cellpose-SAM)
- macOS 15.x Sequoia, Apple Silicon MPS backend

## Pipeline overview

1. `scripts/export_tiles.py` — tile raw stacks into 512×512 crops for annotation
2. Annotate in Cellpose GUI: `python -m cellpose`
3. `notebooks/04_finetuning.ipynb` — fine-tune cpsam on annotations
4. `scripts/run_pipeline.py` — segment all images, measure intensities
5. `notebooks/07_analysis_plots.ipynb` — visualise results

## Data

Raw data stored in `data/raw/YYYYMMDD_condition/`. Not committed to git (too large).
Document acquisition parameters in `data/raw/YYYYMMDD_condition/README.md`.

Input format: `(4, H, W)` uint16 TIFF stacks exported from Zeiss Zen (Airyscan).

## Notes on 16-bit data

**Always apply masks to the original 16-bit arrays for measurement.**
The normalised BFP image used for segmentation must never be used for intensity
quantification. See `src/measure.py` for the correct pattern.
