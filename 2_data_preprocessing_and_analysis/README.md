# Data Preprocessing and Analysis

This folder processes the raw Calabria wildfire dataset into structured training data and produces exploratory analysis plots.

---

## Requirements

**Before running anything**, add the following to `../1_data/raw/`:
- `WarningAreas/` — the warning zone shapefile (`.shp`, `.dbf`, `.shx`, `.prj`)

Some scripts require the original `.h5` dataset which lives on the remote server at `../../Calabria_dataset/InputReteGood/`. These are marked below and must be run on the remote server after cloning the repo there.

---

## Step-by-Step Execution

**Step 1 — Build the processed dataset (remote server required):**
Run `01_warning_areas_analysis.ipynb` first — it produces `cell_zones.parquet` which every subsequent script depends on.
Then run `05_build_yearly_dataset.ipynb` to process all years, followed by `06_merge_yearly_data.ipynb` to merge them into `zone_sequence_merged.csv`. This is the main training dataset.

**Step 2 — Spatial grids (remote server required):**
Run `10_compute_spatial_grids.py` once to generate the precomputed `.npy` fire and temperature grids needed by steps 11 and 12.

**Step 3 — Analysis and plots (local, no remote needed):**
Once the processed files exist, the following can be run in any order:
- `07_eda_fire_analysis.ipynb` — fire distribution, monthly/yearly trends, time series
- `08_eda_climate_analysis.ipynb` — climate variable trends by month and year
- `09_eda_correlation_analysis.ipynb` — correlation between climate and fire counts
- `11_spatial_maps.ipynb` — fire frequency and temperature maps
- `12_burned_area_analysis.ipynb` — burned area statistics per zone

**Exploratory only (remote server required, not needed to reproduce results):**
`02`, `03`, and `04` are single-day exploration notebooks used during development. They are kept for reference but are not part of the main pipeline.

---

## Outputs

| File | Location |
|------|----------|
| `cell_zones.parquet` | `../1_data/processed/` |
| `yearly/*.csv` | `../1_data/processed/yearly/` |
| `zone_sequence_merged.csv` | `../1_data/processed/` |
| `spatial_grids/*.npy` | `../1_data/processed/spatial_grids/` |
| `burned_area/*.csv` | `../1_data/processed/burned_area/` |
| `plots/` | `plots/` |