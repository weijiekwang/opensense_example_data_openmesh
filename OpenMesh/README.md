```markdown
# OpenMesh Dataset

OpenMesh is a wireless backhaul link dataset from NYC Community Mesh Network, formatted to OpenSense v1.1 standards.

## Dataset

- **103 wireless links** in New York City
- **Period:** Oct 2023 - Jul 2024 (8 months)
- **Resolution:** ~10 seconds
- **Format:** NetCDF (OpenSense v1.1)

## Quick Start

```bash
# Clone repository
git clone https://github.com/drorjac/opensense_example_data_openmesh.git
cd opensense_example_data_openmesh/OpenMesh/notebooks

# Setup environment
conda create -n openmesh python=3.11
conda activate openmesh
pip install xarray pandas numpy matplotlib seaborn netCDF4
```

## Download Data

Download OpenMesh dataset from Zenodo:
- **URL:** https://zenodo.org/records/15268341
- Extract `OpenMesh.zip` to `notebooks/data/openmesh/`

## Load Data

```python
import openmesh_data_converter as odc

# Load link data
ds_nyc = odc.transform_jacoby_2025_OpenMesh(
    fn="data/openmesh/OpenMesh.zip",
    extract_path="data/openmesh/"
)

# Load metadata
df_metadata = odc.fix_metadata_to_opensense("data/openmesh/links_metadata.csv")

# Load weather data
weather_data = odc.load_weather_data("data/weather/")
```

## Citation

```
Jacoby, D. et al. (2025). OpenMesh: a quality-controlled dataset of 
wireless backhaul link attenuation in New York City. 
Earth System Science Data Discussions (in review).
https://doi.org/10.5194/essd-2025-238
```
