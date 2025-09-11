# Methodology

## Overview

This document describes the statistical and methodological framework used to generate the China City Mass Index (CMI) interactive map. The analysis implements a hierarchical mismatch detection system to identify how Chinese cities and neighborhoods deviate from global Upper Middle Income Country (UMIC) patterns in terms of material intensity.

## Statistical Framework

### Hierarchical Mixed-Effects Model

The core analysis uses a hierarchical mixed-effects model to account for the nested structure of neighborhoods within cities within countries:

```
log₁₀(M) = α_global + α_country + α_city + β × log₁₀(N) + ε
```

Where:
- `M` = Total built material mass per neighborhood (tonnes)
- `N` = Population per neighborhood (2015)
- `β` = Fixed scaling exponent (0.75, from literature)
- `α_global` = Global intercept
- `α_country` = Country-specific random intercept
- `α_city` = City-specific random intercept (nested within country)
- `ε` = Residual error term

### Random Effects Structure

The model uses nested random effects:
- `(1 | country)` - Country-level variation in material intensity
- `(1 | country:city)` - City-level variation within countries

This structure allows for:
1. Countries to have different baseline material intensities
2. Cities within countries to deviate from their national average
3. Neighborhoods within cities to deviate from their city average

## Key Metrics

### 1. City Mass Index (CMI)

The City Mass Index represents how much a city's material intensity deviates from the UMIC baseline:

```
City CMI = exp(α_global + α_country + α_city)
```

**Interpretation:**
- CMI = 1.0: City matches UMIC baseline
- CMI > 1.0: City has higher material intensity than UMIC baseline
- CMI < 1.0: City has lower material intensity than UMIC baseline

### 2. Neighborhood Multiplier

The neighborhood multiplier shows how individual neighborhoods deviate from their city's mean:

```
Neighborhood Multiplier = Observed Mass / Predicted Mass (from city model)
```

**Interpretation:**
- Multiplier = 1.0: Neighborhood matches city average
- Multiplier > 1.0: Neighborhood above city average
- Multiplier < 1.0: Neighborhood below city average

### 3. Outlier Detection

Neighborhoods are classified as outliers using percentile-based thresholds:
- **Low Outliers**: < 5th percentile (displayed in dark blue)
- **High Outliers**: > 95th percentile (displayed in red)
- **Normal Range**: 5th-95th percentile (color-coded by value)

## Data Processing Pipeline

### 1. Spatial Framework

**H3 Hexagonal Grid System**
- Resolution 6 hexagons (~36 km² average area)
- Provides consistent neighborhood-scale analysis units
- Clips to urban center boundaries from GHSL

**Urban Center Definitions**
- Uses GHS Urban Centres Database (GHSL-UCDB)
- Filters to functionally urban areas
- Excludes rural and peri-urban areas

### 2. Material Intensity Calculations

**Building Materials**

Building material masses are calculated using climate-adjusted intensities:

```
Material Mass = Building Volume × Intensity Factor × Climate Adjustment
```

**Building Classifications:**
- **LW**: Lightweight buildings (<3m height)
- **RS**: Residential single-family (3-12m)
- **RM**: Residential multi-family (12-50m)
- **NR**: Non-residential (3-50m)
- **HR**: High-rise (>50m)

**Climate Adjustments:**
Based on Köppen climate classification:
- Tropical climates: -10% steel, -12% concrete
- Continental climates: +10% steel, +12% concrete
- Temperate climates: Baseline intensities

**Road Materials**

Road material masses use standardized thickness assumptions:
- Asphalt layer: 15cm thickness, 2.3 t/m³ density
- Concrete base: 20cm thickness, 2.4 t/m³ density

### 3. Quality Control

**Data Validation:**
- Remove neighborhoods with zero population
- Remove neighborhoods with zero built mass
- Validate geometric integrity of spatial boundaries
- Check for outliers beyond physical possibilities

**Imputation Methods:**
- Climate data: Assign based on city majority climate
- Missing building data: Use regional building class distributions
- Population data: WorldPop 2015 at 100m resolution

## Country Classification System

### Upper Middle Income Countries (UMIC)

The analysis uses World Bank income classifications as the baseline comparison group:

**UMIC Definition (2024):**
- GNI per capita: $4,516 - $14,005
- Includes major developing economies
- Provides appropriate comparison group for China

**Key UMIC Countries in Analysis:**
- Albania, Algeria, Argentina, Armenia, Azerbaijan
- Belarus, Belize, Bosnia and Herzegovina, Botswana, Brazil
- Bulgaria, China, Colombia, Costa Rica, Cuba
- [Full list in config/country_classification.csv]

### Rationale for UMIC Comparison

1. **Economic Development Level**: China's income level falls within UMIC range
2. **Urbanization Patterns**: Similar urban development trajectories
3. **Construction Technologies**: Comparable building technologies and materials
4. **Statistical Power**: Sufficient sample size for robust analysis

## Model Assumptions and Limitations

### Assumptions

1. **Scaling Relationship**: Material intensity follows power-law scaling with population
2. **Fixed Scaling Exponent**: β = 0.75 based on neighborhood-level scaling literature
3. **Log-Normal Distribution**: Log-transformed variables are approximately normal
4. **Independence**: Observations are independent after accounting for spatial hierarchy
5. **Stationarity**: Relationships are consistent across space (within income groups)

### Limitations

1. **Data Availability**: Limited to cities with sufficient remote sensing coverage
2. **Temporal Mismatches**: Building data (2020), Population data (2015)
3. **Material Intensity Estimates**: Based on European studies, may not fully represent local practices
4. **Urban Boundary Definitions**: GHSL boundaries may not match administrative boundaries
5. **Climate Adjustments**: Simplified climate categories may miss local variations

### Sensitivity Analysis

The methodology includes several robustness checks:

1. **Alternative H3 Resolutions**: Testing with resolutions 5, 6, and 7
2. **Different Scaling Exponents**: Testing β values from 0.6 to 0.9
3. **Outlier Threshold Sensitivity**: Testing 1st-99th vs 5th-95th percentiles
4. **Building Class Aggregations**: Testing different building type groupings

## Uncertainty Quantification

### Sources of Uncertainty

1. **Building Volume Estimates**: ±20-30% accuracy from remote sensing
2. **Material Intensity Factors**: ±15-25% uncertainty in literature values
3. **Population Estimates**: ±10-15% accuracy from WorldPop
4. **Geometric Processing**: <5% uncertainty from spatial operations

### Uncertainty Propagation

Total uncertainty in material mass estimates:
- **Neighborhood Level**: ±30-40%
- **City Level**: ±20-30% (aggregation reduces uncertainty)
- **Relative Comparisons**: ±15-25% (systematic errors partially cancel)

### Statistical Significance

The hierarchical model provides uncertainty estimates for:
- Country-level random effects (confidence intervals)
- City-level random effects (prediction intervals)
- Residual variation (model fit diagnostics)

## Validation and Verification

### Cross-Validation Approaches

1. **Spatial Cross-Validation**: Hold-out cities for model validation
2. **Temporal Validation**: Compare with historical data where available
3. **External Data Comparison**: Cross-reference with national statistics

### Model Diagnostics

1. **Residual Analysis**: Check for spatial autocorrelation and heteroscedasticity
2. **Random Effects Diagnostics**: Validate nested structure assumptions
3. **Influence Analysis**: Identify influential observations and outliers

### Ground-Truth Validation

Limited ground-truth validation using:
- National building material consumption statistics
- City-level construction data (where available)
- Expert knowledge and literature comparison

## Implementation Details

### Computational Considerations

1. **Memory Management**: Chunked processing for large datasets
2. **Parallel Processing**: City-level analysis parallelized where possible
3. **Caching**: Intermediate results cached to avoid recomputation
4. **Optimization**: Spatial operations optimized for performance

### Reproducibility

1. **Seed Values**: Fixed random seeds for reproducible results
2. **Version Control**: All dependencies versioned
3. **Environment Management**: Conda environment specification
4. **Data Provenance**: Complete data lineage documentation

## References

1. Bettencourt, L. M. (2013). The origins of scaling in cities. *Science*, 340(6139), 1438-1441.
2. Haberl, H., et al. (2021). A systematic review of the evidence on decoupling of GDP, resource use and GHG emissions. *Environmental Research Letters*, 16(4), 043002.
3. Krausmann, F., et al. (2017). Global socioeconomic material stocks rise 23-fold over the 20th century. *PNAS*, 114(8), 1880-1885.
4. Zhou, Y., et al. (2022). A global map of urban extent from nightlights. *Environmental Research Letters*, 17(5), 054012.
5. WorldPop (2018). Global high resolution population denominators project. University of Southampton.

---

**Note**: This methodology is based on the research framework developed for the NYU China Grant project on urban scaling and material stock analysis. For technical implementation details, see the code documentation and configuration files.