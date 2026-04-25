"""
One-shot script to generate the three Jupyter notebooks.
Run from project root with any Python that has the `json` stdlib.
"""
import json
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

_id = 0
def _next():
    global _id; _id += 1; return f"cell{_id:03d}"

def md(text: str) -> dict:
    lines = text.split('\n')
    src = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": _next(), "metadata": {}, "source": src}

def code(text: str) -> dict:
    lines = text.split('\n')
    src = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "execution_count": None, "id": _next(),
            "metadata": {}, "outputs": [], "source": src}

def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "cellpose", "language": "python", "name": "cellpose"},
            "language_info": {"name": "python", "version": "3.10.20"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def save(path: str, cells: list):
    Path(path).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False))
    print(f"  wrote {path}")

# =============================================================================
# 00_data_exploration.ipynb
# =============================================================================
nb00 = [

md("""# 00 · Data Exploration

First look at the raw 16-bit Airyscan stacks.
Establishes image dimensions, per-channel signal quality,
autofluorescence distribution, and the bacteria-dense crop coordinates
used for the baseline comparison."""),

code("""import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from src.io import load_stack
from src.preprocess import normalize_for_segmentation"""),

code("""STACK_PATH    = 'data/raw/16bit/Scene-01-20260416-C3M2_Tcol_1-A01Export-01_c1-4_stack.tif'
PIXEL_SIZE_UM = 0.035   # µm/px (from Zen Image Properties → Scaling)

Path('figures/qc').mkdir(parents=True, exist_ok=True)"""),

md("## Load and verify stack"),

code("""stack = load_stack(STACK_PATH)
print(f'Shape : {stack.shape}')
print(f'Dtype : {stack.dtype}')
print(f'Global max: {stack.max()}')
print(f'Image size: {stack.shape[1]*PIXEL_SIZE_UM:.1f} × {stack.shape[2]*PIXEL_SIZE_UM:.1f} µm')"""),

md("""## Per-channel intensity statistics

Raw uint16 values — BFP is the segmentation channel (C1).
GFP and RFP are stress reporters; SYTOX labels host nuclei."""),

code("""CHANNELS = ['BFP (C1)', 'GFP (C2)', 'RFP (C3)', 'SYTOX (C4)']

print(f'{'Channel':<14} {'Min':>7} {'Max':>7} {'Mean':>8} {'Std':>8} {'p1':>7} {'p99':>8}')
print('-' * 65)
for i, name in enumerate(CHANNELS):
    ch = stack[i].astype(float)
    p1, p99 = np.percentile(ch, [1, 99])
    print(f'{name:<14} {ch.min():>7.0f} {ch.max():>7.0f} '
          f'{ch.mean():>8.1f} {ch.std():>8.1f} {p1:>7.0f} {p99:>8.0f}')"""),

md("## 4-channel visualisation"),

code("""fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
cmaps = ['Blues', 'Greens', 'Reds', 'Purples']

for i, (ax, name, cmap) in enumerate(zip(axes, CHANNELS, cmaps)):
    img_norm = normalize_for_segmentation(stack[i])
    ax.imshow(img_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(name, color='white', fontsize=11)
    ax.axis('off')

fig.suptitle('All 4 channels — percentile-stretched for display only',
             color='white', fontsize=11)
plt.tight_layout(pad=0.3)
plt.savefig('figures/qc/00_four_channels.png', dpi=100,
            bbox_inches='tight', facecolor='black')
plt.show()"""),

md("""## Spatial survey — 3×3 grid (BFP channel)

Checks that signal is uniform and autofluorescence hot-spots are identifiable."""),

code("""H, W = stack.shape[1], stack.shape[2]
bfp = stack[0]

n = 3
ys = np.linspace(0, H - 512, n, dtype=int)
xs = np.linspace(0, W - 512, n, dtype=int)

fig, axes = plt.subplots(n, n, figsize=(12, 12), facecolor='black')
for row, y0 in enumerate(ys):
    for col, x0 in enumerate(xs):
        crop = bfp[y0:y0+512, x0:x0+512]
        axes[row][col].imshow(normalize_for_segmentation(crop),
                              cmap='Blues', vmin=0, vmax=1)
        axes[row][col].set_title(f'y={y0} x={x0}', color='white', fontsize=8)
        axes[row][col].axis('off')

fig.suptitle('BFP channel — 3×3 spatial survey (512×512 px each)',
             color='white', fontsize=11)
plt.tight_layout(pad=0.2)
plt.savefig('figures/qc/00_spatial_survey_bfp.png', dpi=100,
            bbox_inches='tight', facecolor='black')
plt.show()"""),

md("## Per-channel intensity histograms (log scale)"),

code("""# Subsample to speed up histogram computation
rng = np.random.default_rng(42)
idx = rng.integers(0, H * W, size=500_000)
rows, cols = np.unravel_index(idx, (H, W))

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor='#0a0a0a')
colors = ['#4488ff', '#44ff88', '#ff4444', '#cc44ff']

for i, (ax, name, color) in enumerate(zip(axes.flat, CHANNELS, colors)):
    ax.set_facecolor('#111111')
    vals = stack[i][rows, cols].astype(float)
    ax.hist(vals, bins=200, color=color, alpha=0.85, log=True)
    p1, p99 = np.percentile(stack[i], [1, 99])
    ax.axvline(p1,  color='white', lw=0.8, ls='--', label=f'p1={p1:.0f}')
    ax.axvline(p99, color='yellow', lw=0.8, ls='--', label=f'p99={p99:.0f}')
    ax.set_title(name, color='white')
    ax.set_xlabel('Raw uint16 value', color='#aaa')
    ax.set_ylabel('Count (log)', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

fig.suptitle('Per-channel intensity histograms (500k pixel subsample, log y)',
             color='white', fontsize=11)
plt.tight_layout()
plt.savefig('figures/qc/00_channel_histograms.png', dpi=100,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()"""),

md("""## Cross-channel correlations

Key diagnostic: autofluorescent fecal debris shows high GFP **and** RFP
simultaneously relative to BFP.
Genuine bacteria show BFP-high, GFP/RFP variable (biological state dependent)."""),

code("""bfp_s = stack[0][rows, cols].astype(float)
gfp_s = stack[1][rows, cols].astype(float)
rfp_s = stack[2][rows, cols].astype(float)
syt_s = stack[3][rows, cols].astype(float)

pairs = [
    (bfp_s, gfp_s, 'BFP', 'GFP', '#44ff88'),
    (bfp_s, rfp_s, 'BFP', 'RFP', '#ff4444'),
    (bfp_s, syt_s, 'BFP', 'SYTOX', '#cc44ff'),
    (gfp_s, rfp_s, 'GFP', 'RFP', '#ffcc44'),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0a0a0a')
for ax, (x, y, xn, yn, col) in zip(axes, pairs):
    ax.set_facecolor('#111')
    ax.hexbin(x, y, gridsize=60, cmap='inferno', bins='log', mincnt=1)
    ax.set_xlabel(xn, color='white')
    ax.set_ylabel(yn, color='white')
    ax.set_title(f'{xn} vs {yn}', color='white')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

fig.suptitle('Cross-channel correlations (500k pixel subsample, log density)',
             color='white', fontsize=11)
plt.tight_layout()
plt.savefig('figures/qc/00_cross_correlations.png', dpi=100,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()"""),

md("""## Bacteria-dense crop preview

Crop used for baseline model comparison: **y=5800, x=3300, size=512**.
Identified as a region with dense bacteria and moderate autofluorescence."""),

code("""CROP_Y, CROP_X, CROP_SIZE = 5800, 3300, 512

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
cmaps = ['Blues', 'Greens', 'Reds', 'Purples']

for i, (ax, name, cmap) in enumerate(zip(axes, CHANNELS, cmaps)):
    crop = stack[i][CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]
    ax.imshow(normalize_for_segmentation(crop), cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'{name}\\n{crop.shape[0]}×{crop.shape[1]} px', color='white', fontsize=10)
    ax.axis('off')

size_um = CROP_SIZE * PIXEL_SIZE_UM
fig.suptitle(f'Bacteria-dense crop — y={CROP_Y} x={CROP_X} | '
             f'{CROP_SIZE}×{CROP_SIZE} px = {size_um:.1f}×{size_um:.1f} µm',
             color='white', fontsize=11)
plt.tight_layout(pad=0.3)
plt.savefig('figures/qc/00_crop_preview.png', dpi=150,
            bbox_inches='tight', facecolor='black')
plt.show()"""),

md("""## Summary notes

| | |
|---|---|
| Image size | (4, 6323, 6344) uint16 |
| Pixel size | 0.035 µm/px (from Zen) |
| Field of view | ~221 × 222 µm |
| BFP dynamic range | p1→p99 values above |
| Autofluorescence | Identifiable as high min(GFP,RFP)/BFP pixels — see cross-correlation |
| Bacteria-dense crop | y=5800, x=3300, 512×512 px |

Next: `02_cellpose_baseline.ipynb` — run cpsam on this crop."""),

]

# =============================================================================
# 02_cellpose_baseline.ipynb
# =============================================================================
nb02 = [

md("""# 02 · Cellpose Baseline — cpsam

Runs **Cellpose-SAM (cpsam)** out-of-the-box on a 512×512 BFP crop.
No fine-tuning. Results feed into `03_model_comparison.ipynb`.

**Environment:** Cellpose 4.x (`cellpose`)

Cellpose 3 models (`cyto3`, `bact_fluor_cp3`) are generated separately with
`scripts/run_cellpose3_baseline.py` in a Cellpose 3.x environment. Do not run
those model names through Cellpose 4.x here; v4 can warn that the requested
model was not found and silently fall back to the default cpsam model.

**Outputs:**
- `data/processed/comparison_patch_bfp.tif` — input tile (shared with Cellpose 3 and Omnipose scripts)
- `data/processed/masks_cpsam.tif`
- `figures/qc/baseline_cellpose_models.png`"""),

code("""import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models
import time

from src.io import load_stack
from src.preprocess import normalize_for_segmentation"""),

code("""STACK_PATH    = 'data/raw/16bit/Scene-01-20260416-C3M2_Tcol_1-A01Export-01_c1-4_stack.tif'
PIXEL_SIZE_UM = 0.035   # µm/px — from Zen Image Properties → Scaling

# Bacteria-dense crop (verified within bounds: 5800+512=6312 ≤ 6323)
CROP_Y, CROP_X, CROP_SIZE = 5800, 3300, 512

# Diameter estimate: short axis ~1.5 µm / pixel size
DIAMETER_PX = 1.5 / PIXEL_SIZE_UM
print(f'Diameter estimate: {DIAMETER_PX:.1f} px  ({1.5} µm / {PIXEL_SIZE_UM} µm/px)')

Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('figures/qc').mkdir(parents=True, exist_ok=True)"""),

md("## Load and crop BFP channel"),

code("""print('Loading stack...')
stack = load_stack(STACK_PATH)
print(f'Stack: {stack.shape} {stack.dtype}  max={stack.max()}')

bfp_full  = stack[0]
crop_raw  = bfp_full[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]
crop_norm = normalize_for_segmentation(crop_raw)  # display only; models use raw intensities

print(f'Crop shape : {crop_norm.shape}')
print(f'Raw range  : [{crop_raw.min()}, {crop_raw.max()}]')

tifffile.imwrite('data/processed/comparison_patch_bfp.tif',
                 crop_raw, compression='lzw')
print('Saved → data/processed/comparison_patch_bfp.tif')"""),

md("""## Run Cellpose-SAM (cpsam)

Default model in Cellpose 4.x. Uses SAM backbone for improved boundary detection."""),

code("""print('\\n── Cellpose-SAM (cpsam) ──')
model_cpsam = models.CellposeModel(gpu=True)   # default = cpsam in 4.x

t0 = time.time()
masks_cpsam, flows_cpsam, _ = model_cpsam.eval(
    crop_raw.astype(np.float32),
    diameter=DIAMETER_PX,
    channels=[0, 0],    # grayscale, no nuclear channel
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)
t_cpsam = time.time() - t0
n_cpsam = masks_cpsam.max()
print(f'cpsam: {n_cpsam} cells in {t_cpsam:.1f}s')

tifffile.imwrite('data/processed/masks_cpsam.tif',
                 masks_cpsam.astype(np.uint16), compression='lzw')
print('Saved → data/processed/masks_cpsam.tif')"""),

md("""## Cellpose 3 models

Run `scripts/run_cellpose3_baseline.py` in a Cellpose 3.x environment to
generate `masks_cyto3.tif` and `masks_bact_fluor_cp3.tif`. They are not run in
this notebook because Cellpose 4.x can fall back to cpsam when asked for CP3
model names."""),

md("## Summary"),

code("""print(f'{"Model":<18} {"Cells":>6} {"Time":>8}')
print(f'{"cpsam":<18} {n_cpsam:>6} {t_cpsam:>7.1f}s')
print('\\nCellpose 3 models: run scripts/run_cellpose3_baseline.py in a Cellpose 3.x env.')
print('\\nOmnipose: run scripts/run_omnipose_baseline.py in the omnipose env next.')"""),

md("## Quick visual (cpsam)"),

code("""fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='black')

def outline_overlay(ax, img, masks, title, n_cells):
    ax.imshow(img, cmap='Blues', vmin=0, vmax=1)
    if masks.max() > 0:
        ax.contour(masks, levels=np.arange(0.5, masks.max() + 0.5),
                   colors='red', linewidths=0.4)
    ax.set_title(f'{title}\\n{n_cells} cells', color='white', fontsize=11)
    ax.axis('off')

outline_overlay(ax, crop_norm, masks_cpsam, 'Cellpose-SAM (cpsam)', n_cpsam)

fig.suptitle(f'Baseline — BFP {CROP_SIZE}×{CROP_SIZE} px | '
             f'y={CROP_Y} x={CROP_X} | diameter={DIAMETER_PX:.0f} px',
             color='white', fontsize=11)
plt.tight_layout(pad=0.3)
plt.savefig('figures/qc/baseline_cellpose_models.png',
            dpi=150, bbox_inches='tight', facecolor='black')
print('Figure saved → figures/qc/baseline_cellpose_models.png')
plt.show()"""),

]

# =============================================================================
# 03_model_comparison.ipynb
# =============================================================================
nb03 = [

md("""# 03 · Model Comparison — cpsam vs cyto3 vs bact_fluor_cp3 vs Omnipose

Loads all saved baseline mask files and produces the final comparison figure.

**Environment:** `cellpose`
**Run after:**
1. `02_cellpose_baseline.ipynb` — generates cpsam masks and the shared input crop
2. `scripts/run_cellpose3_baseline.py` (in a Cellpose 3.x env) — generates cyto3 and bact_fluor_cp3 masks
3. `scripts/run_omnipose_baseline.py` (in `omnipose` env) — generates Omnipose masks

**Outputs:**
- `figures/qc/model_comparison_baseline.png`
- `figures/qc/model_comparison_cell_sizes.png`"""),

code("""import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops"""),

md("## Load all results"),

code("""crop_raw    = tifffile.imread('data/processed/comparison_patch_bfp.tif').astype(float)
masks_cpsam = tifffile.imread('data/processed/masks_cpsam.tif').astype(int)

def load_optional_mask(path, label):
    path = Path(path)
    if path.exists():
        print(f'{label} masks loaded.')
        return tifffile.imread(str(path)).astype(int), True
    print(f'WARNING: {path.name} not found — run scripts/run_cellpose3_baseline.py '
          'in a Cellpose 3.x env first.')
    return np.zeros_like(masks_cpsam), False

masks_cyto3, have_cyto3 = load_optional_mask('data/processed/masks_cyto3.tif', 'cyto3')
masks_bact_cp3, have_bact_cp3 = load_optional_mask(
    'data/processed/masks_bact_fluor_cp3.tif', 'bact_fluor_cp3')

omnipose_path = Path('data/processed/masks_bact_fluor_omni.tif')
legacy_omnipose_path = Path('data/processed/masks_omnipose.tif')
if not omnipose_path.exists() and legacy_omnipose_path.exists():
    omnipose_path = legacy_omnipose_path
if omnipose_path.exists():
    masks_omni = tifffile.imread(str(omnipose_path)).astype(int)
    have_omni  = True
    print('Omnipose masks loaded.')
else:
    masks_omni = np.zeros_like(masks_cpsam)
    have_omni  = False
    print('WARNING: masks_bact_fluor_omni.tif not found — '
          'run scripts/run_omnipose_baseline.py in the omnipose env first.')

lo = np.percentile(crop_raw, 1)
hi = np.percentile(crop_raw, 99)
crop_norm = np.clip((crop_raw - lo) / max(hi - lo, 1), 0, 1)"""),

md("## Per-cell size distributions"),

code("""def cell_areas(masks):
    return [r.area for r in regionprops(masks)]

areas_cpsam = cell_areas(masks_cpsam)
areas_cyto3 = cell_areas(masks_cyto3)
areas_bact_cp3 = cell_areas(masks_bact_cp3)
areas_omni  = cell_areas(masks_omni) if have_omni else []

print(f'{"Model":<18} {"N cells":>8} {"Median area (px²)":>18} {"Mean area (px²)":>16}')
print('-' * 64)
for name, areas in [
    ('cpsam', areas_cpsam),
    ('cyto3', areas_cyto3),
    ('bact_fluor_cp3', areas_bact_cp3),
    ('bact_fluor_omni', areas_omni),
]:
    if areas:
        print(f'{name:<18} {len(areas):>8} '
              f'{np.median(areas):>18.1f} {np.mean(areas):>16.1f}')
    else:
        print(f'{name:<18} {"—":>8}')"""),

md("""## Mask overlay comparison

Row 1: contour overlays on BFP image.
Row 2: raw label maps (each cell a unique colour)."""),

code("""def draw_panel(ax, img, masks, title, color='red'):
    ax.imshow(img, cmap='Blues', vmin=0, vmax=1)
    if masks.max() > 0:
        ax.contour(masks, levels=np.arange(0.5, masks.max() + 0.5),
                   colors=color, linewidths=0.5)
    ax.set_title(f'{title}\\n{masks.max()} cells', color='white', fontsize=11, pad=6)
    ax.axis('off')

fig, axes = plt.subplots(2, 4, figsize=(24, 12), facecolor='black')

draw_panel(axes[0][0], crop_norm, masks_cpsam,  'Cellpose-SAM (cpsam)', '#ff4444')
draw_panel(axes[0][1], crop_norm, masks_cyto3,  'Cellpose cyto3',       '#44ff88')
draw_panel(axes[0][2], crop_norm, masks_bact_cp3, 'Cellpose bact_fluor_cp3', '#44ccff')
if have_omni:
    draw_panel(axes[0][3], crop_norm, masks_omni, 'Omnipose bact_fluor_omni', '#ffcc44')
else:
    axes[0][3].imshow(crop_norm, cmap='Blues')
    axes[0][3].set_title('Omnipose bact_fluor_omni\\n(not yet run)', color='yellow', fontsize=11)
    axes[0][3].axis('off')

for ax, masks, title in zip(axes[1],
                             [masks_cpsam, masks_cyto3, masks_bact_cp3, masks_omni],
                             ['cpsam labels', 'cyto3 labels', 'bact_fluor_cp3 labels', 'bact_fluor_omni labels']):
    ax.imshow(masks, cmap='tab20b', interpolation='nearest')
    ax.set_title(title, color='white', fontsize=11)
    ax.axis('off')

if not have_omni:
    axes[1][3].set_facecolor('black')
    axes[1][3].text(0.5, 0.5, 'Run omnipose script first',
                    ha='center', va='center', color='yellow',
                    transform=axes[1][3].transAxes, fontsize=12)
    axes[1][3].axis('off')

fig.suptitle('Baseline model comparison — no fine-tuning\\n'
             'BFP channel | Red=cpsam  Green=cyto3  Blue=bact_fluor_cp3  Yellow=bact_fluor_omni',
             color='white', fontsize=12, y=0.99)
plt.tight_layout(pad=0.3)
plt.savefig('figures/qc/model_comparison_baseline.png',
            dpi=150, bbox_inches='tight', facecolor='black')
print('Saved → figures/qc/model_comparison_baseline.png')
plt.show()"""),

md("## Cell area histograms"),

code("""fig2, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0a')
ax.set_facecolor('#111111')
bins = np.linspace(0, 400, 60)

if areas_cpsam:
    ax.hist(areas_cpsam, bins=bins, alpha=0.7, color='#ff4444',
            label=f'cpsam (n={len(areas_cpsam)})', density=True)
if areas_cyto3:
    ax.hist(areas_cyto3, bins=bins, alpha=0.7, color='#44ff88',
            label=f'cyto3 (n={len(areas_cyto3)})', density=True)
if areas_bact_cp3:
    ax.hist(areas_bact_cp3, bins=bins, alpha=0.7, color='#44ccff',
            label=f'bact_fluor_cp3 (n={len(areas_bact_cp3)})', density=True)
if areas_omni:
    ax.hist(areas_omni, bins=bins, alpha=0.7, color='#ffcc44',
            label=f'bact_fluor_omni (n={len(areas_omni)})', density=True)

ax.axvspan(8, 300, alpha=0.08, color='white',
           label='Expected bacterial size (8–300 px²)')
ax.set_xlabel('Cell area (px²)', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('Cell size distributions — objects outside 8–300 px² are likely noise or debris',
             color='white', fontsize=10)
ax.legend(framealpha=0.3, labelcolor='white')
ax.tick_params(colors='#aaaaaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333333')

plt.tight_layout()
plt.savefig('figures/qc/model_comparison_cell_sizes.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print('Saved → figures/qc/model_comparison_cell_sizes.png')
plt.show()"""),

]

# =============================================================================
# Write files
# =============================================================================
print("Generating notebooks...")
save('notebooks/00_data_exploration.ipynb', nb00)
save('notebooks/02_cellpose_baseline.ipynb', nb02)
save('notebooks/03_model_comparison.ipynb', nb03)
print("Done.")
