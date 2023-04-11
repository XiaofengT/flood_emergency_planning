import os
import geopandas as gpd
import rasterio
import sys
from pyproj import CRS
from pyproj import Transformer

# Load data into program
background = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'background', 'raster-50k_2724246.tif'))
elevation_ras = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'elevation', 'SZ.asc'))
out_elevation = os.path.join('flood_emergency_planning', 'Material', 'out_elevation.tif')
isle_shape = gpd.read_file('flood_emergency_planning/Material/shape/isle_of_wight.shp')
solent_itn_json_1 = os.path.join('flood_emergency_planning', 'Material', 'itn', 'solent_itn.json')
solent_itn_json = os.path.join('flood_emergency_planning', 'Material', 'itn', 'solent_itn.json')
dataset = rasterio.open(os.path.join('flood_emergency_planning', 'Material', 'elevation', 'SZ.asc'))


def input_point():
    # User input point in British National Grid
    lat_input = input('Please enter latitude: ')
    long_input = input('Please enter longitude: ')
    reference_input = input('Please enter the coordinate system, is it wgs84 or osgb36: ')
    if reference_input == 'wgs84':
        # Generate transformer tool to transform wgs84 coordinates into osgb36
        wsg84 = CRS.from_epsg(4326)
        osgb36 = CRS.from_epsg(27700)
        transformer = Transformer.from_crs(wsg84, osgb36)
        # Transform the coordinates
        input_osgb = transformer.transform(lat_input, long_input)
        lat_osgb = input_osgb[0]
        long_osgb = input_osgb[1]
        print('The transformed user input coordinate in British National Grid is: ' + str(lat_osgb) + ', ' + str(
            long_osgb))
    elif reference_input == 'osgb36':  # If the input is in osgb36, then do not transform
        input_osgb = (float(lat_input), float(long_input))
        lat_osgb = input_osgb[0]
        long_osgb = input_osgb[1]
        print('The user input coordinate in British National Grid is: ' + str(lat_osgb) + ', ' + str(
            long_osgb))
    else:  # Inform the user to try again
        print('Undefined coordinate system, please exit the program and try again')
        sys.exit(0)

    return input_osgb


# Determine if the point is in bounding box
def is_inside_bound(coord):
    if 430000 <= coord[0] <= 465000:
        if 80000 <= coord[1] <= 95000:
            return True
        else:
            return False
    else:
        return False


def main():
    input_osgb = input_point()


if __name__ == '__main__':
    main()

