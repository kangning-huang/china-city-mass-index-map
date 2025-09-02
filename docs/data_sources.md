# Data Sources

This document provides comprehensive information about all data sources used in the China City Mass Index Map generation pipeline.

## Primary Spatial Data Sources

### 1. Urban Centre Boundaries

**Source**: GHS Urban Centres Database (GHS-UCDB)
- **Dataset**: `users/kh3657/GHS_STAT_UCDB2015` (Google Earth Engine)
- **Original Source**: JRC Global Human Settlement Layer
- **Spatial Resolution**: Vector polygons
- **Temporal Coverage**: Based on 2015 settlement patterns
- **Coverage**: Global urban centres with population > 50,000
- **License**: Open access
- **URL**: https://ghsl.jrc.ec.europa.eu/ghs_stat_ucdb2015mt_globe_r2019a.php

**Key Attributes Used:**
- `ID_HDC_G0`: Unique city identifier
- `UC_NM_MN`: Urban centre main name
- `CTR_MN_ISO`: Country ISO code
- `CTR_MN_NM`: Country name
- `GRGN_L1`, `GRGN_L2`: Geographic regions

**Data Quality Notes:**
- High-quality delineation of functional urban areas
- May not align with administrative boundaries
- Some cities may be split into multiple polygons

### 2. Building Heights and Volumes

**Source**: Zhou et al. (2022) Global Building Height Dataset
- **Platform**: Google Earth Engine
- **Dataset ID**: Various building height assets
- **Spatial Resolution**: 500m grid
- **Temporal Coverage**: ~2020
- **Coverage**: Global, urban areas
- **Method**: Spaceborne LiDAR and optical imagery

**Building Classes:**
- **LW**: Lightweight buildings (<3m)
- **RS**: Residential single-family (3-12m)
- **RM**: Residential multi-family (12-50m)  
- **NR**: Non-residential (3-50m)
- **HR**: High-rise buildings (>50m)

**Citation**:
Zhou, Y., et al. (2022). A global map of urban extent from nightlights. *Environmental Research Letters*, 17(5), 054012.

### 3. Population Data

**Source**: WorldPop Global Population Datasets
- **Platform**: Google Earth Engine  
- **Dataset**: WorldPop Global 100m Population
- **Asset ID**: `WorldPop/GP/100m/pop/[YEAR]`
- **Spatial Resolution**: 100m (~1 hectare)
- **Temporal Coverage**: 2000-2020 (using 2015 for consistency)
- **Coverage**: Global
- **Method**: Census-based population redistribution

**Data Quality Notes:**
- High spatial resolution population estimates
- Consistent methodology across countries
- Regular updates and improvements
- Some uncertainty in rural and remote areas

**Citation**:
WorldPop (2018). Global high resolution population denominators project. University of Southampton.

### 4. Road Network Data

**Source**: OpenStreetMap via Google Earth Engine
- **Platform**: Google Earth Engine
- **Dataset**: Various OSM road assets
- **Temporal Coverage**: Varies by region, generally recent
- **Coverage**: Global, with varying completeness
- **Road Classifications**:
  - Highway: Limited access roads
  - Primary: Major roads
  - Secondary: Secondary roads  
  - Tertiary: Tertiary roads
  - Local: Local/residential roads

**Data Quality Notes:**
- Quality varies significantly by region
- Generally better coverage in developed areas
- May miss some rural or newly constructed roads
- Regular updates through community contributions

### 5. Impervious Surface Data

**Source**: GAIA FROM-GLC Impervious Surface
- **Platform**: Google Earth Engine
- **Dataset**: FROM-GLC global land cover
- **Asset ID**: `users/wang_dongyu/FROM-GLC_2015/FROM_GLC_2015_100m_impervious`
- **Spatial Resolution**: 100m
- **Temporal Coverage**: 2015
- **Coverage**: Global
- **Method**: Landsat-based classification

**Purpose**: Used for urban masking and pavement area calculations

### 6. Administrative Boundaries

**Source**: China Administrative Boundaries
- **File**: `data/raw/ChinaAdminProvince.gpkg`
- **Source**: Various Chinese government sources
- **Level**: Provincial boundaries
- **Purpose**: National boundary visualization and context
- **Format**: GeoPackage vector format

## Material Intensity Data Sources

### 1. Building Material Intensities

**Primary Source**: Haberl et al. (2021) - European Building Stock Analysis
- **Materials Covered**: Steel, concrete, wood, other structural materials
- **Building Types**: Residential, commercial, industrial
- **Geographic Scope**: European countries (adapted for global use)
- **Units**: tonnes per m³ of built volume

**Climate Adjustments Based On**:
- Thermal performance requirements
- Structural design standards
- Local building codes and practices
- Köppen climate classification system

**Citation**:
Haberl, H., et al. (2021). A systematic review of the evidence on decoupling of GDP, resource use and GHG emissions. *Environmental Research Letters*, 16(4), 043002.

### 2. Road Material Intensities

**Source**: Engineering Standards and Literature
- **Asphalt Roads**: 
  - Thickness: 15cm standard
  - Density: 2.3 tonnes/m³
- **Concrete Roads**:
  - Base thickness: 20cm standard  
  - Density: 2.4 tonnes/m³

**References**:
- Highway engineering standards
- Construction industry guidelines
- Transportation infrastructure literature

### 3. Climate Classification

**Source**: Köppen-Geiger Climate Classification
- **Dataset**: Global climate zones
- **Resolution**: ~1km grid
- **Categories Used**:
  - A: Tropical
  - B: Arid  
  - C: Temperate
  - D: Continental
  - E: Polar

**Purpose**: Adjust material intensities based on climate requirements

## Economic Classification Data

### Country Income Classifications

**Source**: World Bank Country Classifications
- **File**: `config/country_classification.csv`
- **Updated**: Annual updates from World Bank
- **Classifications**:
  - High income: >$14,005 GNI per capita
  - Upper middle income: $4,516-$14,005
  - Lower middle income: $1,136-$4,515  
  - Low income: <$1,136

**UMIC Countries in Analysis** (Selected Examples):
- Albania (ALB), Algeria (DZA), Argentina (ARG)
- Brazil (BRA), Bulgaria (BGR), China (CHN)
- Colombia (COL), Costa Rica (CRI), Cuba (CUB)
- [Complete list in CSV file]

**URL**: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

## Auxiliary Data Sources

### 1. H3 Spatial Indexing

**Source**: Uber H3 Geospatial Indexing System
- **Library**: h3-py Python library
- **Resolution**: 6 (average area ~36 km²)
- **Purpose**: Consistent spatial units for neighborhood analysis
- **Properties**: Hexagonal grid, hierarchical, global coverage

**Citation**:
Uber Technologies (2018). H3: A hexagonal hierarchical geospatial indexing system.

### 2. Areal Interpolation

**Source**: Tobler Python Library
- **Library**: tobler (PySAL ecosystem)
- **Method**: Area-weighted interpolation
- **Purpose**: Disaggregate data to H3 hexagons
- **Functions**: `h3fy()` for H3 grid generation

## Data Processing and Quality Control

### Temporal Alignment

**Reference Year**: 2015
- **Population**: WorldPop 2015
- **Buildings**: ~2020 (latest available)
- **Roads**: Variable (OSM contributions)
- **Impervious**: FROM-GLC 2015

**Note**: Some temporal misalignment exists but is considered acceptable given data availability and urban change timescales.

### Quality Control Measures

1. **Geometric Validation**:
   - Check polygon validity
   - Fix self-intersections
   - Ensure proper CRS handling

2. **Value Range Validation**:
   - Remove negative populations
   - Flag unrealistic building heights
   - Check material intensity bounds

3. **Spatial Consistency**:
   - Verify neighborhood-city containment
   - Check coordinate reference systems
   - Validate area calculations

4. **Statistical Outlier Detection**:
   - Identify extreme values
   - Flag potential data errors
   - Document assumptions and limitations

## Data Access and Licensing

### Google Earth Engine Assets

- **Access**: Requires GEE account and project
- **Authentication**: OAuth2 or service account
- **Quotas**: Subject to GEE usage limits
- **Licensing**: Various, generally open for research

### Open Data Sources

- **WorldPop**: CC BY 4.0 License
- **OpenStreetMap**: Open Database License
- **World Bank**: Open access policy
- **GHSL**: Open access with attribution

### Derived Data Products

All intermediate and final data products from this analysis are made available under open licenses where possible, subject to upstream licensing requirements.

## Data Citation Requirements

When using this analysis or data products, please cite:

1. **This Analysis**:
   ```
   China City Mass Index Map (2024). 
   NYU China Grant Urban Scaling Project.
   ```

2. **Primary Data Sources**:
   - Individual citations as listed above for each dataset
   - Follow specific attribution requirements for each source

3. **Software Dependencies**:
   - Google Earth Engine
   - H3 Geospatial Indexing System
   - PySAL/Tobler
   - Various Python geospatial libraries

## Known Data Limitations

### Spatial Resolution Mismatches

- Building data: 500m grid
- Population data: 100m grid  
- Roads: Vector (variable resolution)
- Analysis units: H3 hexagons (~36 km²)

### Temporal Coverage Gaps

- Building data: ~2020
- Population baseline: 2015
- Road networks: Variable timing
- Economic classifications: Annual updates

### Geographic Coverage

- Urban bias in building height data
- Variable OSM road completeness
- Limited ground-truth validation data
- Potential systematic errors in certain regions

### Methodological Constraints

- European material intensities applied globally
- Simplified climate adjustment factors
- Fixed scaling relationships assumed
- Limited validation in non-UMIC countries

## Future Data Improvements

### Planned Updates

1. **Higher Resolution Building Data**: Integration of newer Landsat/Sentinel analysis
2. **Improved Material Intensities**: Region-specific calibration where possible  
3. **Enhanced Road Networks**: Integration of additional commercial road datasets
4. **Temporal Extensions**: Multi-year analysis capabilities
5. **Ground-Truth Validation**: Collection of city-specific validation data

### Data Integration Opportunities

1. **National Statistics**: Integration with official construction statistics
2. **Commercial Datasets**: High-resolution building and infrastructure data
3. **Satellite Imagery**: Direct analysis of recent imagery for validation
4. **Social Media**: Crowd-sourced validation and updating

---

**Last Updated**: December 2024  
**Contact**: [Project contact information]  
**Version**: 1.0