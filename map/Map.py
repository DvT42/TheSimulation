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
    BASE_PATH = os.path.dirname(os.path.abspath("__file__"))
    BIOME_LEGEND = {0: ([0, 110, 184, 255], "Sea"),
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
    BIOME_MAP_PATH = BASE_PATH + r"\only_biome_map.png"

    def __init__(self):
        self.points = []
        self.colored_biome_map, self.biome_map = Map.analyze_biome_map(Map.BIOME_MAP_PATH)

        # Adjust plot font sizes
        sns.set(font_scale=1.5)
        sns.set_style("white")

        # Set working dir & get data
        # data = et.data.get_data('spatial-vector-lidar', verbose=True)
        os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

        # Import world boundary shapefile
        worldBound_path = os.path.join("data", "spatial-vector-lidar", "global",
                                       "ne_110m_land", "ne_110m_land.shp")
        self.worldBound = gpd.read_file(worldBound_path)

        # Import graticule & world bounding box shapefile data
        graticule_path = os.path.join("data", "spatial-vector-lidar", "global",
                                      "ne_110m_graticules_all", "ne_110m_graticules_15.shp")
        self.graticule = gpd.read_file(graticule_path)

        bbox_path = os.path.join("data", "spatial-vector-lidar", "global",
                                 "ne_110m_graticules_all", "ne_110m_wgs84_bounding_box.shp")
        self.bbox = gpd.read_file(bbox_path)
    
    def place_points(self, locations: np.ndarray):
        # Turn points into list of x,y shapely points, and translate them from km to m.
        locations = [Point(xy) for xy in (locations * 1000)] 

        # Create geodataframe using the points
        locations = gpd.GeoDataFrame(locations,
                                          columns=['geometry'],
                                          crs="ESRI:54030")
        
        # Reproject point locations to the WGS84 projection
        locations_WGS84 = locations.to_crs(self.worldBound.crs)
        
        self.points = locations_WGS84

    def plot_map(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.imshow(self.colored_biome_map,
                  origin="upper",
                  extent=(-180, 180, -90, 90),
                  alpha=0.5,
                  zorder=0)

        self.bbox.plot(ax=ax,
                       alpha=.1,
                       color='grey')
        self.graticule.plot(ax=ax,
                            color='lightgrey')
        # worldBound.plot(ax=ax,
        #                       alpha=0.5,
        #                       cmap='Greys')

        ax.set(title="World map",
               xlabel="X Coordinates (degrees)",
               ylabel="Y Coordinates (degrees)")

        self.points: gpd.GeoDataFrame
        self.points.plot(ax=ax, markersize=100, color='black')

        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)  # For converting units to scientific ones (1ex).
            axis.set_major_formatter(formatter)

        plt.axis('equal')

        plt.show()

    @staticmethod
    def analyze_biome_map(map_path):
        num_map = imageio.v3.imread(map_path)
        legend = Map.BIOME_LEGEND
        bins = np.array([biome[0] for key, biome in legend.items()], dtype=int)
        processed_map = num_map
        biome_map = np.zeros(shape=(800, 1600), dtype=int)
        for i, line in enumerate(num_map):
            # noinspection PyTypeChecker
            processed_map[i], biome_map[i] = Map.multidimentional_approximation(line, bins)
        return processed_map, biome_map

    @staticmethod
    def multidimentional_approximation(arr, bins):
        distances = np.sqrt(((arr - bins[:, np.newaxis, :]) ** 2).sum(axis=2))
        indexes = np.argmin(distances, axis=0)
        return [bins[i] for i in indexes], indexes


if __name__ == "__main__":
    # Create numpy array of x,y point locations
    add_points = np.array([[-5000,   0],
                           [0,   5000],
                           [1500,   -4500]])
    mymap = Map()
    mymap.place_points(add_points)
    mymap.plot_map()
