import os
import geopandas as gpd
import rasterio

# Load data into program
background = rasterio.open(os.path.join('Material', 'background', 'raster-50k_2724246.tif'))
elevation_ras = rasterio.open(os.path.join('Material', 'elevation', 'SZ.asc'))
out_elevation = os.path.join('Material', 'out_elevation.tif')
isle_shape = gpd.read_file('Material/shape/isle_of_wight.shp')
solent_itn_json_1 = os.path.join('Material', 'itn', 'solent_itn.json')
solent_itn_json = os.path.join('Material', 'itn', 'solent_itn.json')
dataset = rasterio.open(os.path.join('Material', 'elevation', 'SZ.asc'))
