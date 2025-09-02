"""
Visualization utilities for creating interactive maps and plots.
"""

import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)

def get_continuous_color(
    value: float,
    min_val: float,
    max_val: float,
    colormap: str = 'seismic'
) -> str:
    """
    Get continuous color based on normalized value using matplotlib colormap.
    
    Args:
        value (float): Value to map to color
        min_val (float): Minimum value in range
        max_val (float): Maximum value in range
        colormap (str): Matplotlib colormap name
        
    Returns:
        str: Hex color string
    """
    try:
        # Handle invalid inputs
        if pd.isna(value) or value is None:
            return '#808080'  # Grey for NaN values
            
        value = float(value)
        min_val = float(min_val)
        max_val = float(max_val)
        
        if not all(np.isfinite([value, min_val, max_val])):
            return '#808080'  # Grey for infinite values
            
    except (ValueError, TypeError):
        return '#808080'  # Grey for non-numeric values
    
    # Normalize value to [0, 1] range
    if max_val == min_val:
        normalized = 0.5  # Use middle of colormap if all values are the same
    else:
        normalized = (value - min_val) / (max_val - min_val)
        # Clamp to [0, 1] range
        normalized = max(0.0, min(1.0, float(normalized)))
    
    # Get color from matplotlib colormap
    try:
        # Use robust colormap access
        try:
            # Try new matplotlib API first (3.5+)
            if hasattr(matplotlib, 'colormaps'):
                cmap = matplotlib.colormaps.get(colormap)
                if cmap is None:
                    cmap = cm.get_cmap(colormap)
            else:
                # Fall back to older API
                cmap = cm.get_cmap(colormap)
        except:
            # Ultimate fallback - use a default colormap
            cmap = cm.get_cmap('seismic')
            
        rgba_color = cmap(normalized)
        
        # Convert RGBA to hex
        hex_color = mcolors.to_hex(rgba_color)
        return hex_color
        
    except Exception as e:
        logger.warning(f"Error getting color from colormap {colormap}: {e}, using fallback")
        # Fallback to simple color mapping
        if normalized < 0.33:
            return '#0000FF'  # Blue
        elif normalized < 0.67:
            return '#FFFF00'  # Yellow
        else:
            return '#FF0000'  # Red

def get_continuous_multiplier_color(
    value: float, 
    is_low_outlier: bool = False, 
    is_high_outlier: bool = False
) -> str:
    """
    Get continuous color based on log-normal distribution for neighborhood multipliers.
    
    Args:
        value (float): Multiplier value
        is_low_outlier (bool): Whether value is a low outlier
        is_high_outlier (bool): Whether value is a high outlier
        
    Returns:
        str: Hex color string
    """
    # Handle outliers first
    if is_low_outlier:
        return '#0D47A1'  # Dark blue for low outliers (< 5th percentile)
    if is_high_outlier:
        return '#D32F2F'  # Red for high outliers (> 95th percentile)
        
    # Handle invalid inputs
    if pd.isna(value) or value is None:
        return '#808080'  # Gray for NaN
        
    try:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            return '#2166AC'  # Dark blue for invalid/zero values
    except (ValueError, TypeError):
        return '#808080'  # Gray for non-numeric values
    
    # Calculate log10 value for thresholds
    try:
        log_value = np.log10(value)
    except:
        return '#808080'  # Gray for calculation errors
    
    # Define color ranges based on log10 thresholds
    color_ranges = [
        ((-float('inf'), -0.20), '#2166AC'),              # Very low: < 0.63 (blue)
        ((-0.20, -0.15), '#2166AC', '#5AADE8'),          # Blue interpolation
        ((-0.15, -0.05), '#5AADE8', '#95B6E0'),          # Light blue transition
        ((-0.05,  0.00), '#95B6E0', '#D3D3D3'),          # Blue → neutral gray
        (( 0.00,  0.05), '#D3D3D3', '#FFFF99'),          # Neutral gray → yellow
        (( 0.05,  0.15), '#FFFF99', '#FFB84D'),          # Yellow → warm orange
        (( 0.15,  0.20), '#FFB84D', '#FF8C00'),          # Orange deepening
        (( 0.20, float('inf')), '#FF0000')               # High: → red
    ]
    
    # Find the appropriate color range
    for range_def in color_ranges:
        if len(range_def) == 2:  # Single color range
            (lower_thresh, upper_thresh), color = range_def
            if lower_thresh <= log_value < upper_thresh:
                return color
        else:  # Interpolation range
            (lower_thresh, upper_thresh), lower_color, upper_color = range_def
            if lower_thresh <= log_value < upper_thresh:
                # Calculate interpolation factor
                if upper_thresh == lower_thresh:
                    return lower_color
                
                factor = (log_value - lower_thresh) / (upper_thresh - lower_thresh)
                factor = max(0.0, min(1.0, factor))
                
                # Interpolate between colors
                try:
                    # Validate hex color format
                    if not (lower_color.startswith('#') and len(lower_color) == 7 and 
                           upper_color.startswith('#') and len(upper_color) == 7):
                        return lower_color if factor < 0.5 else upper_color
                    
                    # Convert hex to RGB
                    lower_rgb = tuple(int(lower_color[i:i+2], 16) for i in (1, 3, 5))
                    upper_rgb = tuple(int(upper_color[i:i+2], 16) for i in (1, 3, 5))
                    
                    # Linear interpolation
                    interp_rgb = tuple(
                        max(0, min(255, int(lower_rgb[j] + factor * (upper_rgb[j] - lower_rgb[j]))))
                        for j in range(3)
                    )
                    
                    # Convert back to hex
                    return f"#{interp_rgb[0]:02x}{interp_rgb[1]:02x}{interp_rgb[2]:02x}"
                    
                except Exception as interp_e:
                    logger.debug(f"Color interpolation error: {interp_e}")
                    # Fallback to discrete color
                    return lower_color if factor < 0.5 else upper_color
    
    # Final fallback
    if log_value >= 0.2:
        return '#FF0000'  # Red for high values
    else:
        return '#2166AC'  # Dark blue for very low values

def calculate_point_sizes(
    gdf: gpd.GeoDataFrame, 
    min_radius: int = 5, 
    max_radius: int = 25
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Calculate point sizes based on polygon areas for map visualization.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries
        min_radius (int): Minimum circle radius in pixels
        max_radius (int): Maximum circle radius in pixels
        
    Returns:
        tuple: (radii series, areas series, area statistics)
    """
    try:
        # Calculate areas in projected coordinate system for accuracy
        # Use World Mercator as fallback
        gdf_projected = gdf.to_crs('EPSG:3395')  # World Mercator
        areas = gdf_projected.geometry.area / 1_000_000  # Convert to km²
        
        # Handle edge cases
        if len(areas) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float), {}
            
        area_min, area_max = areas.min(), areas.max()
        area_median = areas.median()
        
        # Use square root scaling for better visual perception (area → radius relationship)
        sqrt_areas = np.sqrt(areas)
        sqrt_min, sqrt_max = sqrt_areas.min(), sqrt_areas.max()
        
        # Normalize to radius range
        if sqrt_max == sqrt_min:
            radii = pd.Series([min_radius] * len(areas), index=areas.index)
        else:
            normalized_sqrt = (sqrt_areas - sqrt_min) / (sqrt_max - sqrt_min)
            radii = min_radius + normalized_sqrt * (max_radius - min_radius)
        
        area_stats = {
            'min': area_min,
            'max': area_max, 
            'median': area_median,
            'min_radius': min_radius,
            'max_radius': max_radius
        }
        
        logger.info(f"Calculated point sizes: {area_min:.1f} - {area_max:.1f} km² → {min_radius} - {max_radius} px radius")
        
        return radii, areas, area_stats
        
    except Exception as e:
        logger.error(f"Error calculating point sizes: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), {}

def create_folium_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 5,
    tiles: str = 'CartoDB positron'
) -> folium.Map:
    """
    Create a basic Folium map with specified center and tiles.
    
    Args:
        center_lat (float): Map center latitude
        center_lon (float): Map center longitude
        zoom_start (int): Initial zoom level
        tiles (str): Map tile style
        
    Returns:
        folium.Map: Initialized map
    """
    try:
        if tiles == 'CartoDB positron':
            tile_url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
            attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        else:
            tile_url = tiles
            attr = 'Map data'
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tile_url,
            attr=attr
        )
        
        return m
        
    except Exception as e:
        logger.error(f"Error creating Folium map: {e}")
        return folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

def add_layer_control(m: folium.Map):
    """Add layer control to Folium map."""
    try:
        folium.LayerControl().add_to(m)
        logger.debug("Added layer control to map")
    except Exception as e:
        logger.error(f"Error adding layer control: {e}")

def add_fullscreen_button(m: folium.Map):
    """Add fullscreen button to Folium map."""
    try:
        plugins.Fullscreen().add_to(m)
        logger.debug("Added fullscreen button to map")
    except Exception as e:
        logger.error(f"Error adding fullscreen button: {e}")

def create_color_gradient_stops(
    min_val: float, 
    max_val: float, 
    n_stops: int = 10,
    colormap: str = 'seismic'
) -> List[Tuple[float, str]]:
    """
    Create color gradient stops for legend.
    
    Args:
        min_val (float): Minimum value
        max_val (float): Maximum value
        n_stops (int): Number of color stops
        colormap (str): Matplotlib colormap name
        
    Returns:
        list: List of (percentage, color) tuples
    """
    try:
        color_stops = []
        for i in range(n_stops):
            norm_val = i / (n_stops - 1)
            actual_val = min_val + norm_val * (max_val - min_val)
            color = get_continuous_color(actual_val, min_val, max_val, colormap)
            color_stops.append((norm_val * 100, color))
        
        return color_stops
        
    except Exception as e:
        logger.error(f"Error creating color gradient stops: {e}")
        return [(0, '#808080'), (100, '#808080')]

def save_map_safely(
    m: folium.Map, 
    output_path: Union[str, Path],
    use_temp_location: bool = True
) -> bool:
    """
    Save Folium map to file with error handling.
    
    Args:
        m (folium.Map): Map to save
        output_path (Union[str, Path]): Output file path
        use_temp_location (bool): Whether to use temporary location first
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if use_temp_location:
            # Save to temp location first to avoid sync issues
            temp_path = Path(tempfile.gettempdir()) / output_path.name
            m.save(str(temp_path))
            
            # Move to final location
            shutil.move(str(temp_path), str(output_path))
        else:
            m.save(str(output_path))
        
        # Validate the saved file
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"Map successfully saved to: {output_path}")
            logger.info(f"File size: {file_size_mb:.1f} MB")
            return True
        else:
            logger.error("Map file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error saving map: {e}")
        return False

def create_popup_html(data: Dict[str, Union[str, float]], max_width: int = 350) -> folium.Popup:
    """
    Create HTML popup from data dictionary.
    
    Args:
        data (dict): Data to display in popup
        max_width (int): Maximum popup width
        
    Returns:
        folium.Popup: Formatted popup
    """
    try:
        html_lines = []
        for key, value in data.items():
            # Format key as bold
            key_display = key.replace('_', ' ').title()
            
            # Format value based on type
            if isinstance(value, float):
                if abs(value) >= 1000:
                    value_display = f"{value:,.0f}"
                else:
                    value_display = f"{value:.4f}"
            else:
                value_display = str(value)
            
            html_lines.append(f"<b>{key_display}:</b> {value_display}<br>")
        
        popup_html = "".join(html_lines)
        return folium.Popup(popup_html, max_width=max_width)
        
    except Exception as e:
        logger.error(f"Error creating popup HTML: {e}")
        return folium.Popup("No data available", max_width=max_width)

def create_legend_html(
    title: str,
    color_info: Dict,
    position: str = "bottom-right",
    width: int = 300,
    height: int = 200
) -> str:
    """
    Create HTML for map legend.
    
    Args:
        title (str): Legend title
        color_info (dict): Color information and gradients
        position (str): Legend position on map
        width (int): Legend width in pixels
        height (int): Legend height in pixels
        
    Returns:
        str: HTML string for legend
    """
    try:
        position_styles = {
            "bottom-right": "bottom: 50px; right: 50px;",
            "bottom-left": "bottom: 50px; left: 50px;",
            "top-right": "top: 50px; right: 50px;",
            "top-left": "top: 50px; left: 50px;"
        }
        
        position_style = position_styles.get(position, position_styles["bottom-right"])
        
        html = f'''
        <div style="position: fixed; 
                    {position_style} width: {width}px; height: {height}px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <p style="margin: 0 0 8px 0; font-weight: bold;">{title}</p>
            {color_info.get('gradient_html', '')}
            {color_info.get('labels_html', '')}
            {color_info.get('additional_info', '')}
        </div>
        '''
        
        return html
        
    except Exception as e:
        logger.error(f"Error creating legend HTML: {e}")
        return f'<div style="position: fixed; bottom: 50px; right: 50px;"><p>{title}</p></div>'