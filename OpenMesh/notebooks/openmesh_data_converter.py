"""
OpenMesh Data Converter (odc)
Functions for downloading and transforming OpenMesh NYC dataset
Similar to opensense_data_downloader_and_transformer.py

Author: Dror Jacoby
"""

import os
import urllib.request
import zipfile
from functools import partial
import pandas as pd
import xarray as xr
import numpy as np


# ==============================================================================
# CORE DOWNLOAD FUNCTION
# ==============================================================================

def download_data_file(url, local_path=".", local_file_name=None, print_output=False):
    """
    Download data file from URL

    Parameters
    ----------
    url : str
        URL to download from
    local_path : str
        Local path to store the file
    local_file_name : str, optional
        Custom filename (default: extract from URL)
    print_output : bool
        Whether to print progress

    Returns
    -------
    str
        Path to downloaded file
    """
    if not os.path.exists(local_path):
        if print_output:
            print(f"Creating path {local_path}")
        os.makedirs(local_path)

    if local_file_name is None:
        local_file_name = url.split("/")[-1]

    full_path = os.path.join(local_path, local_file_name)

    if os.path.exists(full_path):
        print(f"File already exists at {full_path}")
        print("Not downloading!")
        return full_path

    if print_output:
        print(f"Downloading from {url}")
        print(f"to {full_path}")

    try:
        urllib.request.urlretrieve(url, full_path)
        if print_output:
            print(f"✓ Download complete!")
        return full_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


# ==============================================================================
# OPENMESH-SPECIFIC FUNCTIONS (OpenSense style with partial)
# ==============================================================================

# OpenMesh download function (using partial like OpenSense)
download_jacoby_2025_OpenMesh = partial(
    download_data_file,
    url="https://zenodo.org/records/15268341/files/OpenMesh.zip",
)


def transform_jacoby_2025_OpenMesh(fn, extract_path, reshape=True):
    """
    Transform OpenMesh dataset to OpenSense format

    Parameters
    ----------
    fn : str
        Path to OpenMesh.zip file
    extract_path : str
        Path to extract files to
    reshape : bool, optional
        If True, reshape to full OpenSense format (cml_id, sublink_id, time).
        If False, keep original structure. Default is True.

    Returns
    -------
    xarray.Dataset
        Formatted OpenMesh dataset with OpenSense-compliant attributes
    """
    print("Extracting OpenMesh.zip...")
    with zipfile.ZipFile(fn) as zfile:
        zfile.extractall(extract_path)

    # Load NetCDF
    nc_path = os.path.join(extract_path, "OpenMesh.nc")
    print(f"Loading {nc_path}...")
    ds = xr.open_dataset(nc_path)

    # Reshape if requested
    if reshape:
        print("Reshaping to OpenSense format...")
        ds = _reshape_to_opensense(ds)

    # Add OpenSense attributes
    ds = add_cml_attributes(ds)

    print("Transformation complete!")
    return ds


def _reshape_to_opensense(ds):
    """
    Internal function: Reshape from (time, sublink_id) to (time, sublink_id, cml_id)

    Parameters
    ----------
    ds : xarray.Dataset
        Original dataset with dimensions (time, sublink_id)

    Returns
    -------
    xarray.Dataset
        Reshaped dataset with dimensions (time, sublink_id, cml_id)
    """
    # Rename sublink_id dimension to cml_id
    ds = ds.rename({'sublink_id': 'cml_id'})

    # Expand dimensions to add sublink_id
    ds = ds.expand_dims({'sublink_id': ['sublink_1']})

    # Transpose to correct order
    ds = ds.transpose('time', 'sublink_id', 'cml_id')

    # Update coordinates that need sublink_id dimension
    for coord in ['frequency', 'polarization']:
        if coord in ds.coords:
            ds[coord] = ds[coord].expand_dims({'sublink_id': ['sublink_1']})

    # Fix polarization: 'Vertical' → 'v', 'Horizontal' → 'h'
    if 'polarization' in ds.coords:
        pol = ds['polarization'].values
        pol = np.where(pol == 'Vertical', 'v', pol)
        pol = np.where(pol == 'Horizontal', 'h', pol)
        ds['polarization'] = (('sublink_id', 'cml_id'), pol)

    # Remove tsl (not reliable)
    if 'tsl' in ds.data_vars:
        ds = ds.drop_vars('tsl')

    return ds


# ==============================================================================
# OPENSENSE ATTRIBUTE FUNCTION
# ==============================================================================

def add_cml_attributes(ds):
    """
    Add OpenSense CML v1.1 compliant attributes to dataset
    Based on official OpenSense conventions

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset

    Returns
    -------
    xarray.Dataset
        Dataset with OpenSense-compliant attributes
    """

    dict_attributes = {
        "time": {
            "long_name": "time_utc",
        },
        "cml_id": {
            "long_name": "commercial_microwave_link_identifier",
        },
        "sublink_id": {
            "long_name": "sublink_identifier",
        },
        "site_0_lat": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_0_latitude",
        },
        "site_0_lon": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_0_longitude",
        },
        "site_1_lat": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_1_latitude",
        },
        "site_1_lon": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_1_longitude",
        },
        "length": {
            "units": "m",
            "long_name": "distance_between_pair_of_antennas",
        },
        "frequency": {
            "units": "MHz",
            "long_name": "sublink_frequency",
        },
        "polarization": {
            "units": "no units",
            "long_name": "sublink_polarization",
        },
        "tsl": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "transmitted_signal_level",
            "sampling": "instantaneous",
        },
        "rsl": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "received_signal_level",
            "sampling": "instantaneous",
        },
    }

    # Apply attributes to existing variables
    ds_vars = list(ds.coords) + list(ds.data_vars)
    for v in ds_vars:
        if v in dict_attributes:
            ds[v].attrs = dict_attributes[v]

    # Set time encoding (xarray handles conversion when saving)
    ds.time.encoding['units'] = "seconds since 1970-01-01 00:00:00"

    return ds


# ==============================================================================
# WEATHER DATA LOADING FUNCTION
# ==============================================================================

def load_weather_data(weather_base_path="data/weather/", print_output=True):
    """
    Load all OpenMesh weather data (airports, NOAA, PWS)

    Parameters
    ----------
    weather_base_path : str
        Base path to weather data directory
    print_output : bool
        Whether to print loading progress

    Returns
    -------
    dict
        Dictionary containing:
        - 'airports': dict of airport weather dataframes
        - 'noaa': dict of NOAA weather dataframes
        - 'pws': dict of PWS weather dataframes
        - 'metadata': weather stations metadata dataframe
    """
    weather_data = {}

    # 1. Airport data
    if print_output:
        print("Loading Airport Weather Data...")

    airport_files = {
        'condition': 'condition_airports.csv',
        'dew_point': 'dew_point_airports.csv',
        'humidity': 'humidity_airports.csv',
        'precip': 'precip_airports.csv',
        'pressure': 'pressure_airports.csv',
        'temperature': 'temperature_airports.csv',
        'wind_direction': 'wind_direction_airports.csv',
        'wind_gust': 'wind_gust_airports.csv',
        'wind_speed': 'wind_speed_airports.csv'
    }

    weather_data['airports'] = {}
    for key, filename in airport_files.items():
        path = os.path.join(weather_base_path, 'airports', filename)
        weather_data['airports'][key] = pd.read_csv(path, index_col=0, parse_dates=True)
        if print_output:
            print(f"  ✓ {key}: {weather_data['airports'][key].shape}")

    # 2. NOAA data
    if print_output:
        print("\nLoading NOAA Weather Data...")

    noaa_files = {
        'precip': 'pcpn_noaa.csv',
        'snow': 'snow_noaa.csv',
        'snow_depth': 'snwd_noaa.csv'
    }

    weather_data['noaa'] = {}
    for key, filename in noaa_files.items():
        path = os.path.join(weather_base_path, 'noaa', filename)
        weather_data['noaa'][key] = pd.read_csv(path, index_col=0, parse_dates=True)
        if print_output:
            print(f"  ✓ {key}: {weather_data['noaa'][key].shape}")

    # 3. PWS (Personal Weather Stations) data
    if print_output:
        print("\nLoading PWS Weather Data...")

    pws_files = {
        'dew_point': 'dew_point_wu.csv',
        'humidity': 'humidity_wu.csv',
        'pressure': 'pressure_wu.csv',
        'temperature': 'temperature_wu.csv',
        'wind_direction': 'wind_direction_wu.csv',
        'wind_gust': 'wind_gust_wu.csv',
        'wind_speed': 'wind_speed_wu.csv',
        'precip': 'precip/precip_wu.csv',
        'precip_accum': 'precip/precip_accum_wu.csv'
    }

    weather_data['pws'] = {}
    for key, filename in pws_files.items():
        path = os.path.join(weather_base_path, 'pws', filename)
        weather_data['pws'][key] = pd.read_csv(path, index_col=0, parse_dates=True)
        if print_output:
            print(f"  ✓ {key}: {weather_data['pws'][key].shape}")

    # 4. Weather metadata
    if print_output:
        print("\nLoading Weather Metadata...")

    weather_metadata_path = os.path.join(weather_base_path, 'weather_metadata.csv')
    weather_data['metadata'] = pd.read_csv(weather_metadata_path)

    if print_output:
        print(f"  ✓ metadata: {weather_data['metadata'].shape}")
        print("\n" + "=" * 60)
        print("Weather Data Loading Complete!")
        print("=" * 60)
        print(f"\nAvailable sources: {list(weather_data.keys())}")
        print(f"Total datasets loaded: {sum(len(v) for v in weather_data.values() if isinstance(v, dict))}")

    return weather_data


# ==============================================================================
# METADATA CORRECTION FUNCTION
# ==============================================================================

def adjust_metadata_to_opensense(metadata_path, output_path=None):
    """
    Fix OpenMesh metadata to match OpenSense conventions

    Parameters
    ----------
    metadata_path : str
        Path to original links_metadata.csv
    output_path : str, optional
        Path to save corrected metadata. If None, saves as
        links_metadata_corrected.csv in the same directory

    Returns
    -------
    pandas.DataFrame
        Corrected metadata DataFrame
    """
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    print("Original columns:", list(df.columns))

    # Fix 1: Rename sublink_id to cml_id
    if 'sublink_id' in df.columns:
        df = df.rename(columns={'sublink_id': 'cml_id'})
        print("✓ Renamed 'sublink_id' → 'cml_id'")

    # Fix 2: Add sublink_id column (all are 'sublink_1' since we only have one direction per CML)
    df['sublink_id'] = 'sublink_1'
    print("✓ Added 'sublink_id' column (all set to 'sublink_1')")

    # Fix 3: Convert polarization to short form (v/h)
    if 'polarization' in df.columns:
        df['polarization'] = df['polarization'].replace({
            'Vertical': 'v',
            'Horizontal': 'h'
        })
        print("✓ Converted polarization to 'v'/'h' format")

    # Fix 4: Remove data columns (rsl, tsl) - metadata should only have link properties
    data_cols = ['rsl', 'tsl']
    cols_to_remove = [col for col in data_cols if col in df.columns]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"✓ Removed data columns: {cols_to_remove}")

    # Reorder columns to match OpenSense convention
    column_order = ['cml_id', 'sublink_id', 'site_0_lat', 'site_0_lon',
                    'site_1_lat', 'site_1_lon', 'length', 'frequency', 'polarization']
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Determine output path
    if output_path is None:
        base_dir = os.path.dirname(metadata_path)
        output_path = os.path.join(base_dir, "links_metadata_corrected.csv")

    # Save corrected metadata
    df.to_csv(output_path, index=False)
    print(f"\n✓ Corrected metadata saved to: {output_path}")

    print("\nCorrected columns:", list(df.columns))
    print(f"Number of links: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())

    return df


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def download_and_extract_openmesh(local_path="data/openmesh/", print_output=True):
    """
    Download and extract OpenMesh dataset (one-step function)

    Parameters
    ----------
    local_path : str
        Local path for download and extraction
    print_output : bool
        Whether to print progress

    Returns
    -------
    str
        Path to extracted OpenMesh.nc file
    """
    # Download
    zip_path = download_jacoby_2025_OpenMesh(
        local_path=local_path,
        print_output=print_output
    )

    if zip_path is None:
        return None

    # Extract
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_path)

    nc_file = os.path.join(local_path, "OpenMesh.nc")

    if os.path.exists(nc_file):
        print(f"✓ Extracted: {nc_file}")
        return nc_file
    else:
        print(f"✗ OpenMesh.nc not found in zip")
        return None


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Test download and load
    print("Testing OpenMesh data converter...")

    # Download
    zip_path = download_jacoby_2025_OpenMesh(
        local_path="data/openmesh/",
        print_output=True
    )

    if zip_path:
        # Transform (with reshape by default)
        ds = transform_jacoby_2025_OpenMesh(
            fn=zip_path,
            extract_path="data/openmesh/",
            reshape=True
        )

        print("\n" + "=" * 60)
        print("OPENMESH DATASET LOADED")
        print("=" * 60)
        print(ds)
        print("\n" + "=" * 60)
        print("Dataset has OpenSense-compliant attributes ✓")
        print("=" * 60)