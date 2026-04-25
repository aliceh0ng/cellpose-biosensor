# cellpose-biosensor

Fine-tuning Cellpose-SAM for segmentation of individual fluorescent bacteria in Airyscan confocal images of mouse colon tissue sections, with per-cell fluorescence intensity measurement.

## Biological setup

Germ-free mice monocolonised with a genetically engineered *E. coli* Nissle 1917 strain carrying a constitutive BFP reporter and two stress-inducible reporters (GFP, RFP). Tissue sections were imaged by Zeiss Airyscan confocal (100×, 5×5 tile scans, 16-bit, ~6300×6300 px per image).

| Channel | Fluorophore | Purpose |
|---------|-------------|---------|
| C1 | BFP (constitutive) | Segmentation input + ratio denominator |
| C2 | GFP (osmotic-stress promoter) | Reporter — OFF in low osmolality |
| C3 | RFP (oxidative-stress promoter) | Reporter — measured as RFP/BFP |
| C4 | SYTOX far-red | Host nuclear stain — exclusion mask |

Pixel size: **0.035 µm/px**. Raw files are `(4, H, W)` uint16 TIFFs exported from Zeiss Zen.

## Goal

Measure per-cell GFP/BFP and RFP/BFP ratios to quantify single-cell osmotic and oxidative stress reporter activity across samples. Segment bacteria using the BFP channel.

## Approach

Iterative human-in-the-loop fine-tuning of Cellpose-SAM on manually corrected masks from domain-specific images (bacteria in non-cleared colon tissue with fecal autofluorescence). The fine-tuned model is applied to all 37 image stacks; intensity measurements are extracted from raw 16-bit data using the generated masks.

## Key challenges

- **Fecal autofluorescence** — broad-spectrum signal bright in BFP, GFP, and RFP simultaneously; filtered post-segmentation using min(GFP, RFP)/BFP ratio
- **Non-cleared tissue** — host cell autofluorescence, out-of-plane blur, heterogeneous background
- **Large stitched images** (~6300×6300 px) — split into a 3×3 or 5×5 grid of patches for model input

## Current results

| Model | Internal val AP@0.5 | External val AP@0.5 (Scene-02) |
|-------|---------------------|-------------------------------|
| Base Cellpose-SAM | 0.641 | 0.624 |
| 5stacks (round 1) | 0.842 | 0.658 |
| 4stacks_5x5_norm (round 2) | **0.724** | **0.720** |

Round 2 (`4stacks_5x5_norm`): 4 training stacks, 5×5 patches (~1265×1265 px), global BFP normalisation, multi-channel annotation review. 75 training patches, 3,913 annotated cells.

## Environments

Two conda environments — keep them separate, they conflict.

### cellpose (main)

```bash
conda env create -f environment.yml   # creates env named cellpose-biosensor
conda activate cellpose               # actual env name used at creation
pip install -e .
```

Verify setup:
```bash
python -c "
from cellpose import models
import torch
model = models.CellposeModel(model_type='cpsam', gpu=True)
print('GPU available:', torch.cuda.is_available() or torch.backends.mps.is_available())
print('device:', next(model.net.parameters()).device)
"
```

**macOS (Apple Silicon):** MPS backend, Cellpose 4.1.1, PyTorch 2.11.0, Python 3.10.20  
**Windows (NVIDIA):** CUDA 12.4 build — use `environment_windows.yml` instead

### omnipose (baseline comparison only)

```bash
conda create -n omnipose python=3.10
pip install omnipose tifffile imagecodecs matplotlib numpy
```

CPU only (no MPS support). Used for `bact_fluor_omni` baseline in notebook 03.

## Pipeline overview

### Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| `00_data_exploration.ipynb` | Image QC, channel statistics, crop selection |
| `01_tiling_preprocessing.ipynb` | Export annotation patches (3×3 or 5×5 grid) |
| `02_cellpose_baseline.ipynb` | Run base cpsam and cyto3 on a representative patch |
| `03_model_comparison.ipynb` | Compare all baseline models (requires omnipose env for step 2) |
| `04_finetuning.ipynb` | Fine-tune cpsam on annotated patches |
| `05_validate.ipynb` | Evaluate fine-tuned model vs base on validation image |
| `06_segmentation_pipeline.ipynb` | Full pipeline: tile → segment → filter → measure → CSV |

### Scripts (for HPC or local GPU runs)

```bash
python scripts/make_splits.py                      # generate data/splits.json (70/15/15)
python scripts/inference.py --split train          # run cpsam inference on training split
python scripts/finetune.py --run-name <name>       # fine-tune on annotated patches
python scripts/pipeline.py --split all             # full pipeline on all images
```

SLURM jobs for Sockeye HPC are in `jobs/`. Edit `--account` and `--mail-user` before submitting.

## Post-segmentation filters (applied in order)

1. **Size filter** — remove objects < 8 px² or > 300 px²
2. **Spectral filter** — remove objects where min(GFP, RFP) / BFP > 0.6 (autofluorescence)
3. **SYTOX exclusion** — remove objects overlapping > 25% with SYTOX+ host nuclei
4. **Background subtraction** — local annular ring (5 px) per cell before computing ratios

## Measurement pipeline

```
raw uint16 stack
      │
      ├─→ BFP → normalize_for_segmentation() → Cellpose-SAM → masks
      │                                                            │
      └────────────────────────────────────────────────────────→ extract_intensities(masks, raw_stack)
                                                                      │
                                                                      └→ gfp_over_bfp, rfp_over_bfp per cell
```

**Always measure from the raw uint16 stack.** Normalised images are for segmentation input only.

## Data

Raw data in `data/raw/16bit/` — read-only, not committed to git.  
Annotated patches: `data/annotations/images/` and `data/annotations/masks/`  
Results (per-cell CSVs): `data/results/`
