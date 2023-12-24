import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import earthpy as et


class Map:
    def __init__(self):
        # Adjust plot font sizes
        sns.set(font_scale=1.5)
        sns.set_style("white")

        # Set working dir & get data
        data = et.data.get_data('spatial-vector-lidar', verbose=True)
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
            formatter.set_scientific(True)  # For converting units to scientific ones (1ex).
            axis.set_major_formatter(formatter)

        plt.axis('equal')

        plt.show()

    @staticmethod
    def analyze_biome_map(map_path):
        num_map = imageio.v3.imread(map_path)
        legend = {0: ([0, 110, 184, 255], "Sea"),
                  1: ([255, 255, 255, 255], "Ice"),
                  2: ([191, 160, 234, 255], "Tundra"),
                  3: ([0, 84, 0, 255], "Taiga"),
                  4: ([154, 37, 7, 255], "Montane"),
                  5: ([69, 177, 69, 255], "Forest"),
                  6: ([0, 255, 0, 255], "Tropical Rainforest"),
                  7: ([255, 255, 0, 255], "Steppe"),
                  8: ([249, 178, 51, 255], "Savanna"),
                  9: ([249, 74, 0, 255], "Desert"),
                  10: ([214, 37, 255, 255], "mediterranean")}
        bins = np.array([biome[0] for key, biome in legend.items()])
        processed_map = num_map
        biome_mapping = np.zeros(shape=(800, 1600))
        for i, line in enumerate(num_map):
            # noinspection PyTypeChecker
            processed_map[i], biome_mapping[i] = Map.multidimentional_approximation(line, bins)

    @staticmethod
    def multidimentional_approximation(arr, bins):
        distances = np.sqrt(((arr - bins[:, np.newaxis, :]) ** 2).sum(axis=2))
        indexes = np.argmin(distances, axis=0)
        return [bins[i] for i in indexes], indexes


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath("__file__"))
    biome_map_path = base_path + r"\map\only biome map.png"
    Map.analyze_biome_map(biome_map_path)
    mymap = Map()
