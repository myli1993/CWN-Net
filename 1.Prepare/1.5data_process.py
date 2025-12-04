import os
import numpy as np
from osgeo import gdal
import scipy.ndimage
from pathlib import Path

# Define paths
input_dir = r"E:\水科院工作汇总\个人成果\论文\聊城德州\data\1.model\image"
output_dir = r"E:\水科院工作汇总\个人成果\论文\聊城德州\data\1.model\image_processed"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to check and interpolate NaN/Inf values
def interpolate_invalid(data):
    invalid = np.isnan(data) | np.isinf(data)
    if np.any(invalid):
        data[invalid] = np.interp(
            np.flatnonzero(invalid),
            np.flatnonzero(~invalid),
            data[~invalid]
        )
    return data

# Function to calculate indices
def calculate_indices(bands, band_names):
    indices = {}
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for q in quarters:
        # Get band indices
        red_idx = band_names.index(f'{q}_Red')
        nir_idx = band_names.index(f'{q}_NIR')
        green_idx = band_names.index(f'{q}_Green')
        
        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        nir = bands[nir_idx].astype(float)
        red = bands[red_idx].astype(float)
        ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
        
        # Calculate NDWI: (Green - NIR) / (Green + NIR)
        green = bands[green_idx].astype(float)
        ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), 0)
        
        # Calculate WBI: NIR / Red
        wbi = np.where(red != 0, nir / red, 0)
        
        indices[f'{q}_NDVI'] = ndvi
        indices[f'{q}_NDWI'] = ndwi
        indices[f'{q}_WBI'] = wbi
    
    return indices

# Function to process a single image
def process_image(input_path, output_path):
    # Open image
    dataset = gdal.Open(input_path)
    if dataset is None:
        print(f"Failed to open {input_path}")
        return
    
    # Get band names and data
    band_names = [dataset.GetRasterBand(i+1).GetDescription() for i in range(dataset.RasterCount)]
    bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(dataset.RasterCount)]
    
    # Resize to 128x128
    resized_bands = []
    for band in bands:
        # Check and interpolate invalid values
        band = interpolate_invalid(band)
        # Resize using scipy
        resized = scipy.ndimage.zoom(band, (128/band.shape[0], 128/band.shape[1]), order=1)
        resized_bands.append(resized)
    
    # Calculate indices
    indices = calculate_indices(resized_bands, band_names)
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_path,
        128, 128,
        len(band_names) + len(indices),
        gdal.GDT_Float32
    )
    
    # Write original bands
    for i, (band, name) in enumerate(zip(resized_bands, band_names)):
        out_band = out_dataset.GetRasterBand(i+1)
        out_band.WriteArray(band)
        out_band.SetDescription(name)
    
    # Write index bands
    for i, (name, data) in enumerate(indices.items(), start=len(band_names)+1):
        out_band = out_dataset.GetRasterBand(i)
        out_band.WriteArray(data)
        out_band.SetDescription(name)
    
    # Copy geotransform and projection
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    
    # Close datasets
    out_dataset.FlushCache()
    dataset = None
    out_dataset = None

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.startswith("ID") and filename.endswith("_2024.tif"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        print(f"Processing {filename}...")
        process_image(input_path, output_path)

print("Processing complete!")