# China City Mass Index Map

This repository contains the complete, reproducible pipeline for generating the China City Mass Index (CMI) interactive map. The map visualizes city- and neighborhood-level material intensity variations of Chinese cities compared to global cities in Upper Middle Income Countries (UMIC).

## 🗺️ Interactive Map

The live map is available at: **https://cmi.kangning-huang.com/**

## 📋 Overview

This project implements a hierarchical mismatch detection system that:

1. **Extracts neighborhood-level data** from Google Earth Engine (building volumes, road networks, population)
2. **Calculates material masses** using climate-adjusted building material intensities
3. **Performs statistical analysis** comparing China to other UMIC countries using mixed-effects models
4. **Generates interactive visualizations** showing city-level CMI and neighborhood-level multipliers

## 🔧 Setup Instructions

### Prerequisites

- Python 3.9+
- Google Earth Engine account and project
- Git

### 1. Environment Setup

Clone and set up the environment:

```bash
git clone <repository-url>
cd china-city-mass-index-map

# Option A: Using conda (recommended)
conda env create -f environment.yml
conda activate china-city-mass-index

# Option B: Using pip
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication

```bash
# Authenticate with Google Earth Engine
earthengine authenticate

# Set your GEE project ID in config/settings.py
# Update GEE_PROJECT_ID = "your-project-id"
```

### 3. Configuration

Update `config/settings.py` with your settings:

```python
# Your Google Earth Engine project ID
GEE_PROJECT_ID = "your-gee-project-id"

# Other settings are pre-configured but can be modified
H3_RESOLUTION = 6  # H3 hexagon resolution
```

## 🚀 Quick Start

### Run Complete Pipeline

Generate the full map from scratch:

```bash
python run_pipeline.py --full-pipeline
```

### Run Individual Steps

Execute specific parts of the pipeline:

```bash
# List available steps
python run_pipeline.py --list-steps

# Run single step
python run_pipeline.py --step hierarchical_analysis
```

### Debug Mode

Test with a smaller dataset (Bangladesh):

```bash
python run_pipeline.py --full-pipeline --debug
```

### Country-Specific Analysis

Process specific countries:

```bash
python run_pipeline.py --full-pipeline --country CHN
```

## 📊 Pipeline Overview

The complete pipeline consists of 8 main steps:

```
1. H3 Neighborhoods     → Create hexagonal neighborhoods for all cities
2. Building Extraction  → Extract building volume data from GEE
3. Road Extraction      → Extract road network data from GEE  
4. Data Merging         → Combine building and infrastructure data
5. Material Calculation → Calculate material masses using intensities
6. City Boundaries      → Generate city boundary geometries
7. Hierarchical Analysis → Perform statistical mismatch detection
8. Map Generation       → Create final interactive map
```

## 📁 Repository Structure

```
china-city-mass-index-map/
├── README.md                          # This file
├── run_pipeline.py                    # Main execution script
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── index.html                         # Final interactive map
├── config/
│   ├── settings.py                    # Configuration parameters
│   └── country_classification.csv     # World Bank income classifications
├── scripts/
│   ├── 01_create_h3_neighborhoods.py  # H3 hexagonal grid generation
│   ├── 02_extract_building_data.py    # Building volume data extraction from GEE
│   ├── 03_extract_road_data.py        # Road network data extraction from GEE
│   ├── 04_merge_infrastructure_data.py # Merge building and road datasets
│   ├── 05_calculate_material_masses.py # Calculate material masses from volumes
│   ├── 06_generate_city_boundaries.py # Generate city boundary geometries
│   ├── 07_hierarchical_mismatch_analysis.py # Statistical analysis and CMI calculation
│   ├── 08_create_interactive_map.py    # Generate final interactive visualization
│   └── utils/
│       ├── __init__.py                # Package initialization
│       ├── gee_utils.py               # Google Earth Engine utilities
│       ├── spatial_utils.py           # Spatial processing functions
│       ├── material_intensity_utils.py # Material calculation utilities
│       └── visualization_utils.py      # Map generation and styling utilities
├── data/
│   ├── raw/                          # Input datasets
│   │   ├── ChinaAdminProvince_compressed.gpkg # China administrative boundaries
│   │   └── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg # Global urban centres
│   └── processed/                    # Generated intermediate files
│       ├── boundaries/               # City and neighborhood boundaries
│       ├── neighborhood_data/        # H3-level extracted data
│       └── merged_data/              # Combined datasets
├── results/
│   ├── statistics/                   # Statistical analysis outputs
│   └── maps/                        # Generated map files
├── docs/
│   ├── methodology.md               # Detailed statistical methodology
│   ├── data_sources.md              # Complete data source documentation
│   └── troubleshooting.md           # Common issues and solutions
└── [Generated outputs]               # Pipeline creates additional files as needed
```

## 📈 Data Sources

### Primary Data Sources

- **Urban Boundaries**: GHS Urban Centres Database (GHSL)
- **Building Volumes**: Zhou et al. (2022) global building height dataset
- **Population**: WorldPop 100m resolution
- **Road Networks**: OpenStreetMap via Google Earth Engine
- **Material Intensities**: Haberl et al. (2021), climate-adjusted

### Country Classifications

- **Income Groups**: World Bank classifications (High, Upper Middle, Lower Middle, Low income)
- **UMIC Countries**: Used as baseline for China comparison

## 🔬 Methodology

### Statistical Framework

The analysis uses a hierarchical mixed-effects model:

```
log10(M) = α_global + α_country + α_city + β × log10(N)
```

Where:
- `M` = Material mass per neighborhood
- `N` = Population per neighborhood  
- `β` = Fixed slope (0.75, from literature)
- `α` terms = Random intercepts at different hierarchical levels

### Key Metrics

- **City CMI**: City-level city-mass-index compared to UMIC baseline
- **Neighborhood Relative CMI**: Neighborhood deviation from city mean
- **Mismatch Detection**: Statistical identification of outliers

## 🎯 Outputs

### Primary Outputs

1. **Interactive HTML Map** (`index.html`)
   - Dual-layer visualization (city points + neighborhood hexagons)
   - Toggleable layers with dynamic legends
   - Popup information for detailed statistics

2. **Statistical Results** (`results/statistics/`)
   - City-level statistics (`china_city_statistics.csv`)
   - Neighborhood-level results (`china_neighborhoods_cmi.csv`)

3. **Spatial Boundaries** (`data/processed/boundaries/`)
   - City boundaries (`all_cities_boundaries.gpkg`)
   - H3 hexagon grids (`all_cities_h3_grids.gpkg`)

### Intermediate Outputs

- Neighborhood-level extractions (building volumes, road areas, population)
- Merged datasets with material mass calculations
- Statistical analysis results and model diagnostics

## 🛠️ Advanced Usage

### Custom Material Intensities

Modify material intensities in `config/settings.py`:

```python
MATERIAL_INTENSITIES = {
    'steel': {
        'HR': {'temperate': 0.120, 'tropical': 0.105, 'default': 0.120}
        # ... other building classes
    }
}
```

### Custom H3 Resolution

Change neighborhood size by adjusting H3 resolution:

```python
H3_RESOLUTION = 7  # Smaller hexagons (more detailed)
# H3_RESOLUTION = 5  # Larger hexagons (less detailed)
```

### Adding New Countries

Update `config/country_classification.csv` with new country codes and income classifications.

## 📚 Documentation

- **[Methodology](docs/methodology.md)**: Detailed statistical methodology and assumptions
- **[Data Sources](docs/data_sources.md)**: Complete data source documentation with citations
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## 🚨 Troubleshooting

### Common Issues

1. **Google Earth Engine Authentication**
   ```bash
   earthengine authenticate
   # Follow the prompts to authenticate
   ```

2. **Memory Issues with Large Countries**
   - Use `--debug` flag for testing
   - Process countries individually
   - Increase chunk sizes in `config/settings.py`

3. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Getting Help

- Check [troubleshooting.md](docs/troubleshooting.md) for detailed solutions
- Review pipeline logs for specific error messages
- Ensure all data sources are accessible

## 📄 Citation

If you use this code or methodology, please cite:

```bibtex
@software{china_city_mass_index_2024,
  title={China City Mass Index Interactive Map},
  author={Kangning Huang},
  year={2025},
  url={https://github.com/[username]/china-city-mass-index-map}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Contact

For questions or issues, please:

- Open a GitHub issue
- Contact: kh3657@nyu.edu

---

**Generated for NYU China Grant project - Urban Scaling and Material Stock Analysis**
