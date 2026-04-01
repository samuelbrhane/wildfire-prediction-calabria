# Imports
import os
import re
import numpy as np
import h5py
import multiprocessing

# Paths
FIRE_BASE_DIR = "../../Calabria_dataset/InputReteGood/Target/"
CLIMATE_BASE_DIR = "../../Calabria_dataset/InputReteGood/Climatic/"
OUTPUT_DIR = "../1_data/processed/spatial_grids"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_fire_year(args):
    # Count fire occurrences per grid cell for one year
    year, fire_base_dir, output_dir = args
    print(f"Processing fire year: {year}")

    year_dir = os.path.join(fire_base_dir, year)
    fire_files = sorted([f for f in os.listdir(year_dir) if f.endswith(".h5")])

    fire_counts = None
    grid_initialized = False

    for fire_file in fire_files:
        fire_path = os.path.join(year_dir, fire_file)
        with h5py.File(fire_path, "r") as h5_file:
            values_table = h5_file["values/table"][:]
            attributes_table = h5_file["attributes/table"][:]

        attr_names = [a[0].decode() for a in attributes_table]
        attr_values = [a[1][0] for a in attributes_table]
        attrs = dict(zip(attr_names, attr_values))

        ncols = int(attrs["ncols"])
        nrows = int(attrs["nrows"])

        if not grid_initialized:
            fire_counts = np.zeros((nrows, ncols), dtype=np.uint16)
            grid_initialized = True

        index_values = values_table["index"].astype(int)
        fire_values = values_table["values_block_0"].flatten().astype(int)

        for idx, v in zip(index_values, fire_values):
            if v == 1:
                row = idx // ncols
                col = idx % ncols
                fire_counts[row, col] += 1

    np.save(os.path.join(output_dir, f"fire_counts_{year}.npy"), fire_counts)
    print(f"Saved fire grid for {year}")
    return year



if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    fire_years = sorted(os.listdir(FIRE_BASE_DIR))
    fire_args = [(year, FIRE_BASE_DIR, OUTPUT_DIR) for year in fire_years]

    print("Computing yearly fire count grids...")
    with multiprocessing.Pool(processes=min(len(fire_args), os.cpu_count())) as pool:
        pool.map(process_fire_year, fire_args)

    # Combine yearly fire grids into total
    fire_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("fire_counts_")])
    fire_grids = [np.load(os.path.join(OUTPUT_DIR, f)) for f in fire_files]
    fire_counts_total = np.sum(fire_grids, axis=0)
    np.save(os.path.join(OUTPUT_DIR, "fire_counts_total.npy"), fire_counts_total)
    print(f"Saved total fire grid — max count: {fire_counts_total.max()}")