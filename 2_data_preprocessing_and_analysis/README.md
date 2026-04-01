# Data Preprocessing and Analysis

This folder processes the raw Calabria wildfire dataset into structured training data and produces exploratory analysis plots.

---

## Requirements

**Before running anything**, add the following to `../1_data/raw/`:
- `WarningAreas/` — the warning zone shapefile (`.shp`, `.dbf`, `.shx`, `.prj`)

Some scripts require the original `.h5` dataset which lives on the remote server at `../../Calabria_dataset/InputReteGood/`. These are marked with 🖥️ and must be run on the remote server after cloning the repo there. Scripts marked with ✅ only need files from `../1_data/processed/` and can be run locally.

---

## Step-by-Step Execution

**01 — `01_warning_areas_analysis.ipynb`** 🖥️
Maps each grid cell to one of the 8 warning zones using the shapefile.
Produces `../1_data/processed/cell_zones.parquet` — required by all subsequent scripts.

**02 — `02_climate_analysis.ipynb`** 🖥️
Loads a sample climate `.h5` file and visualizes the spatial distribution of precipitation, humidity, temperature, and wind across zones for a single day.

**03 — `03_target_analysis.ipynb`** 🖥️
Loads a sample fire target `.h5` file and maps wildfire occurrence across the 8 zones for a single day.

**04 — `04_single_day_pipeline_test.ipynb`** 🖥️
Test notebook that validates the full single-day processing logic: reads one day of fire and climate data and produces an 8-row zone summary. This is a prototype of the logic used in `05`.

**05 — `05_build_yearly_dataset.ipynb`** 🖥️
Runs the full data pipeline for all years in parallel using multiprocessing. For each day and each zone it computes fire cluster counts and climate averages. Saves one CSV per year to `../1_data/processed/yearly/`.

**06 — `06_merge_yearly_data.ipynb`** ✅
Merges all yearly CSVs from `../1_data/processed/yearly/` into a single file `../1_data/processed/zone_sequence_merged.csv`. This is the main dataset used for model training.

**07 — `07_eda_fire_analysis.ipynb`** ✅
EDA plots for fire occurrence across all 8 zones and the full region: daily fire count distributions, monthly totals, yearly totals, and fire season time series. Plots saved to `plots/eda_fire/`.

**08 — `08_eda_climate_analysis.ipynb`** ✅
EDA plots for climate variables (temperature, humidity, precipitation, wind): monthly and yearly averages for all 8 zones and the full region. Plots saved to `plots/eda_climate/`.

**09 — `09_eda_correlation_analysis.ipynb`** ✅
Correlation heatmaps between climate variables and fire counts for each zone. Also saves a CSV with same-day and lag-1 day correlation summary to `plots/eda_correlation/`.

**10 — `10_compute_spatial_grids.py`** 🖥️
Runs in parallel to compute yearly fire count grids and yearly average temperature grids from raw `.h5` files. Combines them into a total fire count grid and a mean temperature grid. Saves all `.npy` files to `../1_data/processed/spatial_grids/`. Must be run before `11` and `12`.

**11 — `11_spatial_maps.ipynb`** ✅
Loads the precomputed `.npy` grids from `../1_data/processed/spatial_grids/` and produces two maps: a fire frequency map and a mean temperature map, both overlaid on the zone boundaries.

**12 — `12_burned_area_analysis.ipynb`** ✅
Loads the precomputed fire count grids and computes burned area statistics per zone across all years. Produces Table 2 with mean and total burned area per zone and saves results to `../1_data/processed/burned_area/`.

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