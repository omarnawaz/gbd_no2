# %%
# Imports
# import rioxarray as rxr
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
import xarray as xr
import numpy as np
import os
import glob
from rasterio.merge import merge

# %%
# Functions


def read_netcdf(filename):
    ds = xr.open_dataset(filename).squeeze()
    ds = ds.set_index(x=["lon"], y=["lat"])
    return ds


def calculate_ratio(X, Y, var):
    return X[var] / Y[var]


def forward_project_lur(year, root):
    # Assign tropomi and LUR filenames
    trop_file = "02_TROPOMI/TROPOMI_NO2_V2.4_griddedon0.1grid_"
    trop_base_filename = f"{root}{trop_file}2019.ncf"
    trop_year_filename = f"{root}{trop_file}{year}.ncf"
    base_filename = f"{root}01_LUR/GlobalNO2LUR_2019/GlobalNO2LUR_2019.tif"
    year_filename = f"{root}01_LUR/GlobalNO2LUR_{year}/GlobalNO2LUR_{year}.tif"
    # Read in TROPOMI data and calculate the ratio
    trop_base = read_netcdf(trop_base_filename)
    trop_year = read_netcdf(trop_year_filename)
    ratio = calculate_ratio(trop_year, trop_base, "no2")
    # Open Base LUR and copy meta data for writing
    src = rasterio.open(base_filename)
    output_meta = src.meta.copy()
    output_meta.update(compress="lzw")
    # Open new year LUR for writing and iteratively loop through windows
    with rasterio.open(year_filename, "w", **output_meta) as wrt:
        for ji, window in src.block_windows(1):
            # Get the lats and lons associated with current window
            wdt, hgt = np.meshgrid(np.arange(window.width), np.arange(window.height))
            rows, cols = hgt + window.row_off, wdt + window.col_off
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            lons, lats = xr.DataArray(np.array(xs)), xr.DataArray(np.array(ys))
            xs = slice(lons.min() - 1, lons.min() + 1)
            ys = slice(lats.min() - 1, lats.min() + 1)
            # Select the ratio values associated with lat lon window and interpolate to LUR resolution
            # Note, missing TROPOMI data are replaced with a ratio of 1 (i.e., the base value propogates)
            ratio_t = ratio.sel(x=xs, y=ys)
            ratio_regrid = ratio_t.interp({"x": lons, "y": lats}).fillna(1)
            # Read in Base LUR subset and scale it with the ratio
            lur_subset = src.read(window=window)
            lur_scaled = np.round(lur_subset[0] * ratio_regrid)
            # Set all negative values to the fill value of -128
            lur_scaled = lur_scaled.where(lur_scaled >= 0, -128)
            # Write scaled lur to new LUR file
            wrt.write(lur_scaled.data, 1, window=window)
            print(f"Finished with {float(lons.min()):0.2f}, {float(lats.min()):0.2f}")

