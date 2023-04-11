import os
import geopandas as gpd
import rasterio
import sys
from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Point

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


# Create the function to determine if the point is on island (task 5)
def is_on_island(point):
    point_x = Point(point[0], point[1])
    result = isle_shape.contains(point_x)
    if result[0]:
        return True
    else:
        return False



def main():
    input_osgb = input_point()
    # If the point is not inside bounds, exit the function
    if not is_inside_bound(input_osgb):
        print('User input point outside of bounding box, please exit the application and try again')
        sys.exit(0)
    # Determine if the point is on the island (it could be outside the bounding box in task 1)
    if not is_on_island(input_osgb):
        print('The user input point is not on the Isle of Wight, please exit the application and try again')
        sys.exit(0)


if __name__ == '__main__':
    main()

