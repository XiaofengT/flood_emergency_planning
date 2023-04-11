import os
import geopandas as gpd
import numpy as np
import rasterio
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt
from rasterio.mask import mask
from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry import LineString
from rtree import index
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from cartopy import crs

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


def buffer(user_input):
    # Create a point based on the input coordinates
    point = Point(user_input[0], user_input[1])
    # Create buffer of 5km
    five_km_buffer = point.buffer(5000)
    # Create new GeoDataframe in Geopandas
    buffer_gpd = gpd.GeoDataFrame()
    # Create geometry column and add insert our 5km buffer
    buffer_gpd['geometry'] = None
    buffer_gpd.loc[0, 'geometry'] = five_km_buffer
    buffer_gpd.loc[0, 'location'] = 'user input'
    return five_km_buffer, buffer_gpd


def highest_point(buffer_area):
    # Mask rasterio based on user input point's buffer
    masked_elevation = rasterio.mask.mask(elevation_ras, buffer_area.geometry)
    # Load the masked result into numpy array for calculating max elevation
    elevation_array = np.array(masked_elevation[0])
    max_elev = np.amax(elevation_array)
    print('The max elevation within 5km of the user input point is: ' + str(max_elev) + ' meters.')

    # use the where function to get the location of the max elevation value in the numpy array
    max_pos = np.where(elevation_array == np.amax(elevation_array))
    # print('Tuple of arrays returned : ', max_pos)
    print('List of coordinates of maximum value in Numpy array : ')
    # zip the 2 arrays to get the exact coordinates
    coords = list(zip(max_pos[1], max_pos[2]))
    # print the coordinates in the array to a list
    for cord in coords:
        print('the max elevation value is found here in the array: ' + str(cord))
    # Extract the rows and columns
    pos_row = coords[0][0]
    pos_col = coords[0][1]

    # The raster grid extends 45000 in x-axis and 25000 in y-axis, from bottom to top
    # The elevation array extends 9000 columns and 5000 rows, with a grid size of 5, from top to bottom
    highest_x = (pos_col + 1) * 5 + 425000
    highest_y = (5000 - pos_row) * 5 + 75000
    highest_pt = Point(highest_x, highest_y)
    print('The highest point has coordinate in osgb36 (roughly): ' + str(list(highest_pt.coords)))
    return highest_pt


# Find the node closest to the user's input point and the highest point in the range
def nearest_node(user, highest):
    user_point = Point(user[0], user[1])
    # Check whether the user input is overlapped with the highest point. If so, exit the program
    if user_point == highest:
        print('Your location is safe and you do not need to move.')
        sys.exit()
    # Read itn json file
    with open(solent_itn_json_1, 'r') as j:
        solent_itn = json.load(j)
    # Create an index with the default construction
    idx = index.Index()
    # Insert nodes into the index
    nodes = solent_itn['roadnodes'].items()
    for i, node in enumerate(nodes):
        idx.insert(i, (node[1]['coords'][0], node[1]['coords'][1]), str(node[0]))
    # Creates two empty strings to store the nearest nodes of the user input and the highest point
    near_to_user = ''
    near_to_highest = ''
    # Find the nearest node in index and assign it to the string
    for i in idx.nearest((user_point.x, user_point.y), 1, objects='raw'):
        near_to_user = i, solent_itn['roadnodes'][i]['coords']
    for i in idx.nearest((highest.x, highest.y), 1, objects='raw'):
        near_to_highest = i, solent_itn['roadnodes'][i]['coords']
    return near_to_user, near_to_highest


def shortest_path(start, end, buffer_area):
    # read json ITN
    with open(solent_itn_json, 'r') as f:
        solent_itn = json.load(f)

    # clip json ITN based on user location buffer
    road_links = solent_itn['roadlinks']
    road_nodes = solent_itn['roadnodes']
    # delete_links = []
    # for link in road_links:
    #     # find links which are out of the 5 km buffer
    #     if not LineString(road_links[link]['coords']).intersects(buffer_area):
    #         delete_links.append(link)
    # # delete links
    # for delete_link in delete_links:
    #     road_links.pop(delete_link)

    # import and read raster data
    dataset2 = dataset.read(1)

    # create network graph
    g = nx.DiGraph()
    for link in road_links:
        # get row and column of the start point based on coordinates
        start_row, start_col = dataset.index(road_nodes[road_links[link]['start']]['coords'][0],
                                             road_nodes[road_links[link]['start']]['coords'][1])
        # get height of the start point based on its row and column
        start_height = dataset2[start_row][start_col]
        end_row, end_col = dataset.index(road_nodes[road_links[link]['end']]['coords'][0],
                                         road_nodes[road_links[link]['end']]['coords'][1])
        end_height = dataset2[end_row][end_col]
        # get height difference between start_node and end_node
        diff_h = end_height - start_height

        # calculate the sum ascent according to each segments in link
        link_nodes = road_links[link]['coords']
        link_node_a = link_nodes[0]
        sum_ascent = 0
        # iterate to get height differences of each segments
        for link_node_b in link_nodes[1:]:
            a_row, a_col = dataset.index(link_node_a[0], link_node_a[1])
            a_height = dataset2[a_row][a_col]
            b_row, b_col = dataset.index(link_node_b[0], link_node_b[1])
            b_height = dataset2[b_row][b_col]
            ascent = b_height - a_height
            if ascent > 0:
                sum_ascent += ascent
            link_node_a = link_node_b

        # based on distance and ascent, calculate weights
        length = road_links[link]['length']
        weight_start_end = length / 5000 * 60 + sum_ascent / 10
        weight_end_start = length / 5000 * 60 + (sum_ascent - diff_h) / 10
        # add edges with weights
        g.add_edge(road_links[link]['start'], road_links[link]['end'], fid=link + '_1', weight=weight_start_end)
        g.add_edge(road_links[link]['end'], road_links[link]['start'], fid=link + '_2', weight=weight_end_start)

    # get shortest path using dijkstra
    result = single_source_dijkstra(g, source=start, target=end, weight='weight')
    weight = result[0]
    path = result[1]
    print(weight, path)

    links = []  # this list will be used to populate the feature id (fid) column
    geom = []  # this list will be used to populate the geometry column

    # create shortest path geometry
    first_node = path[0]
    for node in path[1:]:
        # print(g.edges[first_node, node]['fid'])
        link_fid = g.edges[first_node, node]['fid']
        # remove fid suffix '_1', '_2'
        link_fid = link_fid[:len(link_fid) - 2]
        links.append(link_fid)
        geom.append(LineString(road_links[link_fid]['coords']))
        first_node = node

    # draw shortest path in map
    shortest_path_gpd = gpd.GeoDataFrame({'fid': links, 'geometry': geom})
    # shortest_path_gpd.plot()
    # plt.show()

    return weight, shortest_path_gpd


# manipulate the background with palette
def background_img(tiff):
    back_array = tiff.read(1)
    palette = np.array([value for key, value in tiff.colormap(1).items()])
    background_image = palette[back_array]
    bounds = tiff.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return background_image, extent


# crop the elevation raster by the buffer
def elevation_buffer(buffer_crop):
    if os.path.exists(out_elevation):
        os.remove(out_elevation)
    out_img, out_transform = mask(elevation_ras, buffer_crop.geometry, crop=True)
    out_meta = elevation_ras.meta.copy()
    out_meta.update({'height': out_img.shape[1], 'width': out_img.shape[2], 'transform': out_transform})
    with rasterio.open(out_elevation, 'w', **out_meta) as dest:
        dest.write(out_img)
    clipped = rasterio.open(out_elevation)
    array_elevation = clipped.read(1)
    array_elevation_masked = np.ma.masked_where(array_elevation == 0, array_elevation)
    return array_elevation_masked


def new_extent(input_osgb):
    elevation_extent = [input_osgb[0] - 5000, input_osgb[0] + 5000, input_osgb[1] - 5000, input_osgb[1] + 5000]
    display_extent = [input_osgb[0] - 10000, input_osgb[0] + 10000, input_osgb[1] - 10000, input_osgb[1] + 10000]
    return elevation_extent, display_extent


def plotting(basemap, basemap_extent, elevation_out, elevation_extent, display_extent, shortest_route, input_osgb, link,
             time):
    # set the size of figure
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB())
    # set the title
    plt.title('Evacuation Route to the Highest Point on the Isle of Wight', size=5, fontweight='bold')
    # show the background with respective extent
    ax.imshow(basemap, origin='upper', extent=basemap_extent, zorder=0)
    # show the elevation layer
    ele = ax.imshow(elevation_out, zorder=2, origin='upper', extent=elevation_extent, alpha=0.5, cmap='terrain')
    # plot the evacuation route
    shortest_route.plot(ax=ax, edgecolor='blue', linewidth=1, zorder=2, label='evacuation route')
    plt.plot([input_osgb[0], link[0][1][0]], [input_osgb[1], link[0][1][1]], 'blue', linewidth=1, zorder=2)
    # plot the starting point and ending point
    ax.plot(input_osgb[0], input_osgb[1], 'ro', markersize=1, label='starting point')
    ax.plot(link[1][1][0], link[1][1][1], 'go', markersize=1, label='ending point')
    # show the color-bar of elevation layer
    colorbar = plt.colorbar(ele, shrink=0.75)
    colorbar.ax.tick_params(labelsize=5)
    colorbar.set_label('elevation(m)', labelpad=-15, y=1.05, rotation=0, size=5)
    # plot a scale bar
    plt.plot([input_osgb[0] - 8000, input_osgb[0] - 6000], [input_osgb[1] - 8000, input_osgb[1] - 8000], 'black',
             linewidth=1.5, zorder=3)
    plt.text(input_osgb[0] - 7400, input_osgb[1] - 7850, '2km', size=3)
    # plot a north arrow
    plt.arrow(input_osgb[0] - 8000, input_osgb[1] + 6500, 0, 1500, width=0.005, head_width=500)
    plt.text(input_osgb[0] - 8500, input_osgb[1] + 5500, 'N')
    # add estimated time
    t = plt.text(input_osgb[0], input_osgb[1] - 9000, ('Estimated travel time: ' + str(round(time, 1)) + ' minutes'),
                 size=3)
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
    # add a legend on map
    plt.legend(prop={'size': 3})
    ax.set_extent(display_extent, crs=crs.OSGB())
    plt.show()


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

    print('Application processing...')
    # Create buffer, and find link closest to the user input point and highest elevation point
    five_km_buffer, buffer_gpd = buffer(input_osgb)
    highest_pt = highest_point(buffer_gpd)
    link = nearest_node(input_osgb, highest_pt)
    # find shortest path, with the starting node and ending node
    start = link[0][0]
    end = link[1][0]
    # start: nearest ITN node to user
    # end: nearest ITN to highest point
    # buffer_area: polygon with a 5km radius from the user location
    time, shortest_route = shortest_path(start, end, five_km_buffer)
    # Plot the output
    elevation_extent, display_extent = new_extent(input_osgb)
    background_image, background_extent = background_img(background)
    elevation_crop = elevation_buffer(buffer_gpd)
    plotting(background_image, background_extent, elevation_crop, elevation_extent, display_extent, shortest_route,
             input_osgb, link, time)


if __name__ == '__main__':
    main()
