import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import earthpy as et

# Adjust plot font sizes
sns.set(font_scale=1.5)
sns.set_style("white")

# Set working dir & get data
data = et.data.get_data('spatial-vector-lidar')
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# Import world boundary shapefile
worldBound_path = os.path.join("data", "spatial-vector-lidar", "global",
                               "ne_110m_land", "ne_110m_land.shp")
worldBound = gpd.read_file(worldBound_path)

# Create numpy array of x,y point locations
add_points = np.array([[-105.2519,   40.0274],
                       [10.75,   59.95],
                       [2.9833,   39.6167]])

# Turn points into list of x,y shapely points
city_locations = [Point(xy) for xy in add_points]

# Create geodataframe using the points
city_locations = gpd.GeoDataFrame(city_locations,
                                  columns=['geometry'],
                                  crs=worldBound.crs)
city_locations.head(3)

# Import graticule & world bounding box shapefile data
graticule_path = os.path.join("data", "spatial-vector-lidar", "global",
                              "ne_110m_graticules_all", "ne_110m_graticules_15.shp")
graticule = gpd.read_file(graticule_path)

bbox_path = os.path.join("data", "spatial-vector-lidar", "global",
                         "ne_110m_graticules_all", "ne_110m_wgs84_bounding_box.shp")
bbox = gpd.read_file(bbox_path)

# Reproject the data
worldBound_robin = worldBound.to_crs('+proj=robin')
graticule_robin = graticule.to_crs('+proj=robin')
bbox_robin = bbox.to_crs('+proj=robin')

# Reproject point locations to the Robinson projection
city_locations_robin = city_locations.to_crs(worldBound_robin.crs)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

bbox_robin.plot(ax=ax,
                   alpha=.1,
                   color='grey')
graticule_robin.plot(ax=ax,
                        color='lightgrey')
worldBound_robin.plot(ax=ax,
                      cmap='Greys')

ax.set(title="World map (robinson)",
       xlabel="X Coordinates (meters)",
       ylabel="Y Coordinates (meters)")

city_locations_robin.plot(ax=ax, markersize=100, color='springgreen')

for axis in [ax.xaxis, ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)

plt.axis('equal')

plt.show()