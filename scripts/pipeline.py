#!/usr/bin/env python3
"""
scripts/pipeline.py
Full segmentation and measurement pipeline for whole image stacks.

For each image: split → segment (fine-tuned or base cpsam) → stitch →
size filter → spectral filter → SYTOX exclusion → measure → CSV.

Reads model path from models/finetuned/latest_model_path.txt if available;
falls back to base cpsam.

Skips images whose CSV already exists — delete to reprocess.

The `--split` argument uses the stack-level groups from `data/splits.json`.
Those splits are independent of the internal patch-level validation subset used
inside `finetune.py`.
Running the pipeline on `train` stacks is valid for generating outputs and
checking behavior on seen data, but it should not be treated as unbiased
generalization performance. Use `val` for iteration and `test` for final
held-out evaluation.

Usage
-----
# Test on one image first
python scripts/pipeline.py --single FILENAME.tif

# Run on all stack-level splits
python scripts/pipeline.py --split all

# Run only on the held-out stack-level test split
python scripts/pipeline.py --split test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from cellpose import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io      import load_stack, split_image, stitch_masks, save_masks
from src.segment import size_filter, spectral_filter, sytox_exclusion
from src.measure import extract_intensities

RAW_DIR     = PROJECT_ROOT / 'data' / 'raw' / '16bit'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
MASKS_DIR   = PROJECT_ROOT / 'data' / 'processed'
SPLITS_FILE = PROJECT_ROOT / 'data' / 'splits.json'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_ROWS          = 3
N_COLS          = 3
PIXEL_SIZE_UM   = 0.035
DIAMETER_PX     = round(1.5 / PIXEL_SIZE_UM)   # 43 px
FLOW_THRESH     = 0.4
CELLPROB_THRESH = 0.0

MIN_AREA      = 8
MAX_AREA      = 300
AUTOFL_THRESH = 0.6
SYTOX_OVERLAP = 0.25
RING_WIDTH    = 5

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_model():
    path_file = PROJECT_ROOT / 'models' / 'finetuned' / 'latest_model_path.txt'
    if path_file.exists():
        model_path = Path(path_file.read_text().strip())
        if model_path.exists():
            log.info(f'Fine-tuned model: {model_path.name}')
            return models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    log.info('No fine-tuned model found — using base cpsam')
    return models.CellposeModel(gpu=True)


def process_stack(stack_path: Path, model) -> None:
    stem     = stack_path.stem
    csv_out  = RESULTS_DIR / f'{stem}.csv'
    mask_out = MASKS_DIR   / f'{stem}_masks.tif'

    if csv_out.exists():
        log.info(f'  Skip {stem} (CSV exists — delete to reprocess)')
        return

    log.info(f'  Loading...')
    stack   = load_stack(stack_path)
    H, W    = stack.shape[1], stack.shape[2]
    patches = split_image(stack[0], n_rows=N_ROWS, n_cols=N_COLS)
    log.info(f'  {stack.shape}  →  {N_ROWS}×{N_COLS} patches')

    for p in patches:
        t0 = time.time()
        masks, _, _ = model.eval(
            p['tile'].astype(np.float32),
            diameter=DIAMETER_PX,
            channels=[0, 0],
            flow_threshold=FLOW_THRESH,
            cellprob_threshold=CELLPROB_THRESH,
        )
        p['mask'] = masks
        log.info(f'    r{p["row"]}c{p["col"]}  {int(masks.max())} cells  '
                 f'{time.time()-t0:.1f}s')

    masks_raw = stitch_masks(patches, H, W)
    n0 = int(masks_raw.max())

    masks_sz    = size_filter(masks_raw,  min_area=MIN_AREA, max_area=MAX_AREA)
    n1 = int(masks_sz.max())
    masks_sp    = spectral_filter(masks_sz, bfp=stack[0], gfp=stack[1], rfp=stack[2],
                                  max_autofl_score=AUTOFL_THRESH)
    n2 = int(masks_sp.max())
    masks_final = sytox_exclusion(masks_sp, sytox=stack[3], max_overlap=SYTOX_OVERLAP)
    n3 = int(masks_final.max())
    log.info(f'  Filters: raw={n0} → size={n1} → spectral={n2} → sytox={n3} '
             f'(removed {n0-n3})')

    save_masks(masks_final, mask_out)

    log.info(f'  Measuring intensities...')
    df = extract_intensities(masks_final, stack, image_id=stem,
                             local_bg=True, ring_width=RING_WIDTH)
    df.to_csv(csv_out, index=False)
    log.info(f'  Saved {len(df)} cells → {csv_out.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--split',  choices=['train', 'val', 'test', 'all'])
    group.add_argument('--single', metavar='FILENAME')
    args = parser.parse_args()

    model = load_model()
    log.info(f'diameter={DIAMETER_PX} px  normalize=True (auto-adjust saturation)')

    if args.single:
        path = RAW_DIR / args.single
        if not path.exists():
            raise FileNotFoundError(path)
        log.info(f'\n[1/1] {args.single}')
        process_stack(path, model)
    else:
        if not SPLITS_FILE.exists():
            raise FileNotFoundError(
                f'{SPLITS_FILE} not found — run scripts/make_splits.py first')
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        names = (splits['train'] + splits['val'] + splits['test']
                 if args.split == 'all' else splits[args.split])
        log.info(f'Processing {len(names)} images  (split={args.split!r})')
        for i, name in enumerate(names, 1):
            log.info(f'\n[{i}/{len(names)}] {name}')
            process_stack(RAW_DIR / name, model)

    log.info('\nAll done.')


if __name__ == '__main__':
    main()
