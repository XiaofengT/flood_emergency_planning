# Flood Emergency Planning

## Table of Contents
1. Introduction
2. Tasks
3. Material

## 1. Introduction
Extreme flooding is expected on the Isle of Wight and the authority in charge of planning the emergency response is advising everyone to proceed by foot to the nearest high ground. 
To support this process, the emergency response authority wants you to develop a software to quickly advise people of the quickest route that they should take to walk to the highest point of land within a 5km radius.

## 2. Tasks
### 2.1 Task 1: User Input
The application should ask the user to input their current location as a British
National Grid coordinate (easting and northing). Then, it should test whether
the user is within a box (430000, 80000) and (465000, 95000). If the input coordinate is outside this box, inform the user and quit the application. This is done
because the elevation raster provided to you extends only from (425000, 75000)
to (470000, 100000) and the input point must be at least 5km from the edge of
this raster.
### 2.2 Task 2: Highest Point Identification
Identify the highest point within a 5km radius from the user location.
To successfully complete this task you could (1) use the window function in
rasterio to limit the size of your elevation array. If you do not use this window
you may experience memory issues; or, (2) use a rasterised 5km buffer to clip
an elevation array. Other solutions are also accepted. Moreover, if you are not
capable to solve this task you can select a random point within 5km of the user.
### 2.3 Task 3: Nearest Integrated Transport Network
Identify the nearest Integrated Transport Network (ITN) node to the user and
the nearest ITN node to the highest point identified in the previous step. To
successfully complete this task you could use r-trees.
### 2.4 Task 4: Shortest Path
Identify the shortest route using Naismith’s rule from the ITN node nearest to
the user and the ITN node nearest to the highest point.
Naismith’s rule states that a reasonably fit person is capable of waking at 5km/hr and that an additional minute is added for every 10 meters of climb
(i.e., ascent not descent).
To successfully complete this task you could calculate the weight iterating
through each link segment. Moreover, if you are not capable to solve this task
you could (1) approximate this algorithm by calculating the weight using only
the start and end node elevation; (2) identify the shortest distance from the
node nearest the user to the node nearest the highest point using only links in
the ITN.
To test the Naismith’s rule, you can use (439619, 85800) as a starting point.
### 2.5 Task 5: Map Plotting
Plot a background map 20km x 20km of the surrounding area. You are free to
use either a 1:50k Ordnance Survey raster (with internal color-map). Overlay a
transparent elevation raster with a suitable color-map. Add the user’s starting
point with a suitable marker, the highest point within a 5km buffer with a
suitable marker, and the shortest route calculated with a suitable line. Also,
you should add to your map, a color-bar showing the elevation range, a north
arrow, a scale bar, and a legend.
### 2.6 Task 6: Extend the Region
The position of the user is restricted to a region in where the user must be
more than 5km from the edge of the elevation raster. Write additional code to
overcome this limitation

## 3. Material
The data consists of:
1. a shape file delimiting the island;
2. two shape files defining the roads;
3. a JSON file defining the ITN graph;
4. a raster file defining the elevation, and;
5. a raster file to be used as a background map.