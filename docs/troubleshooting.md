# Troubleshooting Guide

This guide helps resolve common issues when setting up and running the China City Mass Index Map pipeline.

## Table of Contents

- [Environment Setup Issues](#environment-setup-issues)
- [Google Earth Engine Problems](#google-earth-engine-problems)
- [Memory and Performance Issues](#memory-and-performance-issues)
- [Data Processing Errors](#data-processing-errors)
- [Visualization Issues](#visualization-issues)
- [Pipeline Execution Problems](#pipeline-execution-problems)

## Environment Setup Issues

### 1. Package Installation Failures

**Problem**: Pip install fails for geospatial packages
```bash
ERROR: Failed building wheel for fiona
```

**Solutions**:

**Option A**: Use conda for geospatial packages
```bash
conda create -n china-cmi python=3.9
conda activate china-cmi
conda install -c conda-forge geopandas rasterio fiona shapely pyproj
pip install -r requirements.txt
```

**Option B**: Install system dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev libspatialindex-dev
pip install -r requirements.txt
```

**Option C**: Use pre-built wheels
```bash
pip install --find-links https://girder.github.io/large_image_wheels GDAL
pip install -r requirements.txt
```

### 2. ImportError for Earth Engine

**Problem**: 
```python
ImportError: No module named 'ee'
```

**Solution**:
```bash
pip install earthengine-api
# OR
conda install -c conda-forge earthengine-api
```

### 3. H3 Library Issues

**Problem**:
```python
ImportError: No module named 'h3'
```

**Solution**:
```bash
pip install h3
# For conda:
conda install -c conda-forge h3-py
```

## Google Earth Engine Problems

### 1. Authentication Issues

**Problem**: 
```
ee.EEException: Please use Earth Engine authenticate to obtain credentials
```

**Solution**:
```bash
earthengine authenticate
```

Follow the authentication flow in your browser and paste the authorization code.

### 2. Project ID Not Set

**Problem**:
```
ee.EEException: Earth Engine not initialized. Please call ee.Initialize()
```

**Solutions**:

**Option A**: Update config file
```python
# In config/settings.py
GEE_PROJECT_ID = "your-project-id"
```

**Option B**: Set environment variable
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

**Option C**: Initialize explicitly
```python
import ee
ee.Initialize(project="your-project-id")
```

### 3. Asset Access Denied

**Problem**:
```
ee.EEException: Asset 'users/kh3657/GHS_STAT_UCDB2015' not found or access denied
```

**Solutions**:
1. Verify asset path in Google Earth Engine Code Editor
2. Request access from asset owner
3. Use alternative public datasets:
   ```python
   # Alternative urban areas dataset
   cities = ee.FeatureCollection("JRC/GHSL/P2016/UCDB/v1")
   ```

### 4. Quota Exceeded

**Problem**:
```
ee.EEException: User memory limit exceeded
```

**Solutions**:
1. Enable chunked processing:
   ```python
   # In config/settings.py
   CHUNK_SIZE = 1000  # Reduce chunk size
   ```

2. Use debug mode for testing:
   ```bash
   python run_pipeline.py --debug
   ```

3. Process individual countries:
   ```bash
   python run_pipeline.py --country CHN
   ```

## Memory and Performance Issues

### 1. Out of Memory Errors

**Problem**:
```
MemoryError: Unable to allocate array
```

**Solutions**:

**Option A**: Increase system memory or use cloud computing
**Option B**: Reduce processing batch size
```python
# In config/settings.py
PROCESSING_BATCH_SIZE = 500  # Reduce from default
```

**Option C**: Use chunked processing
```python
# Process cities in smaller batches
cities_chunks = [cities[i:i+100] for i in range(0, len(cities), 100)]
```

### 2. Slow H3 Grid Generation

**Problem**: H3 grid creation takes very long time

**Solutions**:
1. Use lower H3 resolution for testing:
   ```python
   H3_RESOLUTION = 5  # Instead of 6
   ```

2. Enable parallel processing:
   ```python
   from multiprocessing import Pool
   with Pool() as pool:
       results = pool.map(generate_h3_grids, city_list)
   ```

3. Use spatial indexing:
   ```python
   gdf.sindex  # Build spatial index before operations
   ```

### 3. Large File Sizes

**Problem**: Generated files are too large to handle

**Solutions**:
1. Use compression:
   ```python
   gdf.to_file("output.gpkg", driver="GPKG", compression="DEFLATE")
   ```

2. Filter unnecessary columns:
   ```python
   essential_cols = ['h3index', 'geometry', 'population', 'mass']
   gdf[essential_cols].to_file("output.gpkg")
   ```

3. Use data type optimization:
   ```python
   gdf['population'] = gdf['population'].astype('float32')
   ```

## Data Processing Errors

### 1. Missing Data Files

**Problem**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/...'
```

**Solutions**:
1. Check data directory structure:
   ```bash
   ls -la data/raw/
   ```

2. Download required data files:
   - China boundaries: Download from official sources
   - Country classification: Included in repository

3. Create required directories:
   ```bash
   mkdir -p data/raw data/processed results/statistics
   ```

### 2. CRS/Projection Issues

**Problem**:
```
ValueError: cannot transform unprojected geometries
```

**Solutions**:
1. Set CRS explicitly:
   ```python
   gdf = gdf.set_crs('EPSG:4326')
   ```

2. Reproject before operations:
   ```python
   gdf_proj = gdf.to_crs('EPSG:3857')  # Web Mercator
   ```

3. Check CRS consistency:
   ```python
   print(f"GDF CRS: {gdf.crs}")
   print(f"Other GDF CRS: {other_gdf.crs}")
   ```

### 3. Invalid Geometries

**Problem**:
```
TopologyException: Input geom is invalid
```

**Solutions**:
1. Fix geometries automatically:
   ```python
   gdf['geometry'] = gdf.geometry.buffer(0)
   ```

2. Use make_valid:
   ```python
   from shapely.validation import make_valid
   gdf['geometry'] = gdf.geometry.apply(make_valid)
   ```

3. Filter invalid geometries:
   ```python
   gdf = gdf[gdf.geometry.is_valid]
   ```

## Visualization Issues

### 1. Map Not Displaying

**Problem**: HTML map file opens but shows blank/white page

**Solutions**:
1. Check browser console for JavaScript errors
2. Ensure all data was processed correctly:
   ```python
   print(f"Cities: {len(city_gdf)}")
   print(f"Neighborhoods: {len(neighborhood_gdf)}")
   ```

3. Test with smaller dataset:
   ```bash
   python run_pipeline.py --debug
   ```

### 2. Legend Not Showing

**Problem**: Map displays but legend is missing or malformed

**Solutions**:
1. Check HTML structure in generated file
2. Verify color calculations:
   ```python
   # Test color function
   test_color = get_continuous_color(1.0, 0.5, 2.0, 'seismic')
   print(f"Test color: {test_color}")
   ```

3. Update CSS in visualization utils if needed

### 3. Performance Issues in Browser

**Problem**: Map is slow to load or interact with

**Solutions**:
1. Reduce data density:
   ```python
   # Sample neighborhoods for performance
   neighborhoods_sample = neighborhoods.sample(frac=0.1)
   ```

2. Simplify geometries:
   ```python
   gdf['geometry'] = gdf.geometry.simplify(tolerance=0.001)
   ```

3. Use clustering for dense areas:
   ```python
   from folium.plugins import MarkerCluster
   marker_cluster = MarkerCluster().add_to(map)
   ```

## Pipeline Execution Problems

### 1. Script Imports Failing

**Problem**:
```python
ImportError: No module named 'config.settings'
```

**Solutions**:
1. Run from correct directory:
   ```bash
   cd china-city-mass-index-map
   python run_pipeline.py --help
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

### 2. Step Dependencies Missing

**Problem**: Later pipeline steps fail because earlier outputs missing

**Solutions**:
1. Check intermediate files exist:
   ```bash
   ls -la data/processed/
   ls -la results/statistics/
   ```

2. Run pipeline in order:
   ```bash
   python run_pipeline.py --step h3_neighborhoods
   python run_pipeline.py --step hierarchical_analysis
   ```

3. Use full pipeline for first run:
   ```bash
   python run_pipeline.py --full-pipeline
   ```

### 3. Permission Errors

**Problem**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
1. Check file permissions:
   ```bash
   chmod +w output_directory
   ```

2. Run with appropriate permissions:
   ```bash
   sudo python run_pipeline.py  # Only if necessary
   ```

3. Change output directory:
   ```python
   # In config/settings.py
   BASE_DIR = "/tmp/china-cmi"  # Temporary directory
   ```

## Getting Additional Help

### 1. Enable Debug Logging

Add to your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Check System Information

```python
import sys, platform, geopandas, pandas
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"GeoPandas: {geopandas.__version__}")
print(f"Pandas: {pandas.__version__}")
```

### 3. Minimal Working Example

Create a test script to isolate issues:
```python
# test_minimal.py
import sys
sys.path.append('.')

from config.settings import *
from utils.gee_utils import initialize_gee

print("Testing GEE initialization...")
success = initialize_gee()
print(f"GEE Status: {'OK' if success else 'FAILED'}")

print("Testing file paths...")
ensure_directories()
print("Directories created successfully")
```

### 4. Environment Debugging

```bash
# Check conda environment
conda list

# Check pip packages  
pip list

# Check Python modules
python -c "import ee, geopandas, h3, folium; print('All imports successful')"
```

### 5. Common Environment Setup

If you're still having issues, try this clean setup:

```bash
# Remove existing environment
conda remove -n china-cmi --all

# Create fresh environment
conda create -n china-cmi python=3.9
conda activate china-cmi

# Install core geospatial packages via conda
conda install -c conda-forge geopandas folium matplotlib seaborn pandas numpy

# Install remaining packages via pip
pip install earthengine-api geemap h3 tobler tqdm statsmodels scipy

# Test installation
python -c "import ee, geemap, h3, geopandas, folium; print('Success!')"
```

## Still Need Help?

If you're still experiencing issues:

1. **Check the Issues**: Look at GitHub issues for similar problems
2. **Create New Issue**: Include:
   - Complete error message
   - Python version and platform
   - List of installed packages (`pip freeze`)
   - Minimal code to reproduce the problem
3. **Contact**: Reach out via the contact methods in README.md

---

**Note**: This troubleshooting guide covers the most common issues. For specific technical problems, please refer to the documentation of individual libraries (GeoPandas, Earth Engine, etc.) or create an issue in the repository.