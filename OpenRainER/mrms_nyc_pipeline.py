"""Utilities to download NOAA MRMS data for the New York City area and convert it to NetCDF.

This module builds a simple pipeline that can be used from Python scripts or
Jupyter notebooks.  The pipeline is intentionally explicit so that it can be
adapted for different MRMS products or spatial domains.

Example
-------
>>> from datetime import datetime
>>> from OpenRainER.mrms_nyc_pipeline import download_mrms_series
>>> download_mrms_series(
...     product="MRMS_GaugeCorr_QPE_01H",
...     start_time=datetime(2024, 7, 1),
...     end_time=datetime(2024, 7, 3),
...     output_folder="./data/mrms_nyc",
... )

Prerequisites
-------------
* The MRMS catalog is published at https://noaa-mrms-pds.s3.amazonaws.com
  and is openly accessible via HTTPS.
* Reading MRMS GRIB2 files requires ``cfgrib`` (which depends on ``eccodes``)
  or an equivalent GRIB decoder.  The examples in this module assume
  ``xarray`` + ``cfgrib`` are available.
"""
from __future__ import annotations

import gzip
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import requests
import xarray as xr

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Simple representation of a rectangular geographic area."""

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Return a boolean mask showing which coordinates fall in the box."""

        return (
            (lon >= self.min_lon)
            & (lon <= self.max_lon)
            & (lat >= self.min_lat)
            & (lat <= self.max_lat)
        )


NYC_BOROUGH_BOUNDING_BOX = BoundingBox(
    min_lon=-74.3,
    max_lon=-73.4,
    min_lat=40.45,
    max_lat=41.0,
)
"""A loose bounding box that covers all five New York City boroughs."""


MRMS_BASE_URL = "https://noaa-mrms-pds.s3.amazonaws.com"


class MRMSDownloadError(RuntimeError):
    """Raised when a MRMS object cannot be downloaded."""


@dataclass(frozen=True)
class MRMSProduct:
    """Configuration describing a MRMS product."""

    name: str
    interval_minutes: int

    def filename(self, timestamp: datetime) -> str:
        return f"{self.name}_{timestamp:%Y%m%d-%H%M%S}.grib2.gz"

    def candidate_paths(self, timestamp: datetime) -> Sequence[str]:
        """Generate the most common object keys for MRMS archives.

        MRMS switched directory layouts a few times.  The candidates try the
        different layouts so the pipeline works regardless of the day being
        queried.
        """

        filename = self.filename(timestamp)
        date_path = f"{timestamp:%Y/%m/%d}"
        return (
            f"{self.name}/{filename}",
            f"{self.name}/{date_path}/{filename}",
            f"MRMS/{self.name}/{date_path}/{filename}",
        )


DEFAULT_PRODUCT = MRMSProduct("MRMS_GaugeCorr_QPE_01H", interval_minutes=60)


def daterange(start: datetime, end: datetime, step: timedelta) -> Iterator[datetime]:
    """Yield datetimes from ``start`` up to and including ``end``."""

    current = start
    while current <= end:
        yield current
        current += step


def _ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download(url: str, output_path: Path) -> Path:
    LOGGER.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code != 200:
            raise MRMSDownloadError(
                f"Failed to download {url} (status={response.status_code})"
            )
        with output_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return output_path


def _download_grib(product: MRMSProduct, timestamp: datetime, workdir: Path) -> Path:
    """Download and unpack a MRMS GRIB2 archive into ``workdir``.

    Returns the path to the extracted ``.grib2`` file.
    """

    gz_path = workdir / product.filename(timestamp)
    grib_path = gz_path.with_suffix("")  # strip the .gz extension

    for key in product.candidate_paths(timestamp):
        url = f"{MRMS_BASE_URL}/{key}"
        try:
            _download(url, gz_path)
            break
        except MRMSDownloadError as exc:
            LOGGER.debug("Unable to fetch %s: %s", url, exc)
    else:
        raise MRMSDownloadError(
            f"Unable to find {product.name} file for {timestamp.isoformat()}"
        )

    LOGGER.debug("Decompressing %s", gz_path)
    with gzip.open(gz_path, "rb") as gz_file, grib_path.open("wb") as grib_file:
        grib_file.write(gz_file.read())

    return grib_path


def _subset_dataset(
    dataset: xr.Dataset,
    bounding_box: BoundingBox,
    drop_na: bool = True,
) -> xr.Dataset:
    lon = dataset["longitude"]
    lat = dataset["latitude"]
    mask = bounding_box.contains(lon, lat)
    subset = dataset.where(mask)
    if drop_na:
        subset = subset.dropna(dim="y", how="all")
        subset = subset.dropna(dim="x", how="all")
    return subset


def _load_grib(grib_path: Path) -> xr.Dataset:
    backend_kwargs = {
        "filter_by_keys": {"typeOfLevel": "surface"},
    }
    return xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)


def _convert_to_netcdf(
    grib_path: Path,
    output_path: Path,
    bounding_box: BoundingBox,
) -> Path:
    LOGGER.info("Converting %s to NetCDF", grib_path)
    with _load_grib(grib_path) as ds:
        subset = _subset_dataset(ds, bounding_box)
        subset.to_netcdf(output_path)
    return output_path


def download_mrms_series(
    product: str = DEFAULT_PRODUCT.name,
    start_time: datetime = None,
    end_time: datetime = None,
    output_folder: os.PathLike[str] | str = "./mrms_output",
    bounding_box: BoundingBox = NYC_BOROUGH_BOUNDING_BOX,
    aggregate: bool = True,
    cleanup: bool = True,
) -> Path:
    """Download a time series of MRMS data and return the NetCDF path.

    Parameters
    ----------
    product:
        MRMS product name (for example ``"MRMS_GaugeCorr_QPE_01H"``).
    start_time, end_time:
        Inclusive timestamps defining the desired data window.  The default is
        the most recent product interval if either value is ``None``.
    output_folder:
        Where to store the intermediate and final files.  A ``netcdf``
        subdirectory containing individual files is always created.  When
        ``aggregate`` is ``True`` an additional NetCDF file combining the
        series is produced in the root of ``output_folder``.
    bounding_box:
        Geographic bounding box used to clip the MRMS grid to the NYC region.
    aggregate:
        When ``True`` (default), concatenates the individual timesteps into a
        single multi-time NetCDF.
    cleanup:
        Remove temporary files (downloaded ``.gz`` and ``.grib2`` files) once
        their NetCDF counterparts have been written.
    """

    product_cfg = MRMSProduct(product, interval_minutes=DEFAULT_PRODUCT.interval_minutes)
    output_root = Path(output_folder)
    workdir = output_root / "tmp"
    netcdf_dir = output_root / "netcdf"
    _ensure_folder(workdir)
    _ensure_folder(netcdf_dir)

    if start_time is None or end_time is None:
        now = datetime.utcnow()
        snapped = now - timedelta(minutes=now.minute % product_cfg.interval_minutes)
        start_time = start_time or snapped
        end_time = end_time or snapped

    timestep = timedelta(minutes=product_cfg.interval_minutes)
    netcdf_paths: List[Path] = []

    for timestamp in daterange(start_time, end_time, timestep):
        LOGGER.info("Processing %s", timestamp.isoformat())
        grib_path = _download_grib(product_cfg, timestamp, workdir)
        netcdf_path = netcdf_dir / f"{product_cfg.name}_{timestamp:%Y%m%d%H%M}.nc"
        _convert_to_netcdf(grib_path, netcdf_path, bounding_box)
        netcdf_paths.append(netcdf_path)

        if cleanup:
            gz_path = grib_path.with_suffix(".gz")
            if gz_path.exists():
                gz_path.unlink()
            grib_path.unlink(missing_ok=True)

    if aggregate:
        LOGGER.info("Merging %d NetCDF files", len(netcdf_paths))
        dataset = xr.open_mfdataset([str(p) for p in netcdf_paths], combine="by_coords")
        aggregate_path = output_root / (
            f"{product_cfg.name}_{start_time:%Y%m%d%H%M}_{end_time:%Y%m%d%H%M}.nc"
        )
        dataset.to_netcdf(aggregate_path)
        dataset.close()
        return aggregate_path

    return netcdf_paths[-1]


__all__ = [
    "BoundingBox",
    "NYC_BOROUGH_BOUNDING_BOX",
    "download_mrms_series",
]
