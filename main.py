import os
import geopandas as gpd
import rasterio

# Load data into program
background = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'background', 'raster-50k_2724246.tif'))
elevation_ras = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'elevation', 'SZ.asc'))
out_elevation = os.path.join('flood_emergency_planning', 'Material', 'out_elevation.tif')
isle_shape = gpd.read_file('flood_emergency_planning/Material/shape/isle_of_wight.shp')
solent_itn_json_1 = os.path.join('flood_emergency_planning', 'Material', 'itn', 'solent_itn.json')
solent_itn_json = os.path.join('flood_emergency_planning', 'Material', 'itn', 'solent_itn.json')
dataset = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'elevation', 'SZ.asc'))
