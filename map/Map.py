import os

import earthpy as et
import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import Point


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
    BIOME_MAP_PATH = BASE_PATH + r"\map" + r"\only_biome_map.png"

    def __init__(self, visual):
        self.points = gpd.GeoDataFrame(columns=['geometry'],
                                       crs="ESRI:54001")
        self.colored_biome_map, self.biome_map = Map.analyze_biome_map()

        # Adjust plot font sizes
        sns.set(font_scale=1.5)
        sns.set_style("white")

        # Set working dir & get data
        # data = et.data.get_data('spatial-vector-lidar', verbose=True)

        # Import world boundary shapefile
        worldBound_path = os.path.join(et.io.HOME, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                       "ne_110m_land", "ne_110m_land.shp")
        self.worldBound = gpd.read_file(worldBound_path)

        # Import graticule & world bounding box shapefile data
        graticule_path = os.path.join(et.io.HOME, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                      "ne_110m_graticules_all", "ne_110m_graticules_15.shp")
        self.graticule = gpd.read_file(graticule_path)

        bbox_path = os.path.join(et.io.HOME, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                 "ne_110m_graticules_all", "ne_110m_wgs84_bounding_box.shp")
        self.bbox = gpd.read_file(bbox_path)

        if visual:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))

    def get_biome(self, coordinates):
        return self.biome_map[tuple(np.flip(coordinates, 0))]

    @staticmethod
    def get_surroundings(matrix, coordinates, dtype=int):
        coordinates = tuple(np.flip(coordinates, 0))

        starty = coordinates[0] - 1
        endy = coordinates[0] + 1
        startx = coordinates[1] - 1
        endx = coordinates[1] + 1

        if dtype == int:
            area = np.zeros((3, 3), dtype=int)
        else:
            area = np.empty((3, 3), dtype=dtype)

        if startx == -1:
            if starty == -1:
                area[1:, 1:] = matrix[:2, :2]
                area[1:, 0] = matrix[:2, -1]
            elif endy == 800:
                area[:2, 1:] = matrix[-2:, :2]
                area[:2, 0] = matrix[-2:, -1]
            else:
                area[:, 1:] = matrix[starty:endy + 1, :2]
                area[:, 0] = matrix[starty:endy + 1, -1]
        elif endx == 1600:
            if starty == -1:
                area[1:, :2] = matrix[:2, -2:]
                area[1:, -1] = matrix[:2, 0]
            elif endy == 800:
                area[:2, :2] = matrix[-2:, -2:]
                area[:2, -1] = matrix[-2:, 0]
            else:
                area[:, :2] = matrix[starty:endy + 1, -2:]
                area[:, -1] = matrix[starty:endy + 1, 0]
        elif starty == -1:
            area[1:] = matrix[:2, startx:endx + 1]
        elif endy == 800:
            area[:2] = matrix[-2:, startx:endx + 1]
        else:
            area = matrix[starty: endy + 1,
                                  startx: endx + 1]

        return area

    def convert_points(self, locations: np.ndarray):
        # Turn points into list of x,y shapely points, and translate them from km to m.
        wgs84_locs = (locations - (800, 400)) / (800, 400) * (180, 90)
        points = [Point(xy) for xy in wgs84_locs]


        # Create geodataframe using the points
        gpds = gpd.GeoDataFrame(points,
                                columns=['geometry'],
                                crs="ESRI:54001")

        # Reproject point locations to the WGS84 projection
        # locations_WGS84 = locations.to_crs(self.worldBound.crs)

        # self.points = gpds
        return gpds

    def plot_map(self):
        self.ax.imshow(self.colored_biome_map,
                  origin="upper",
                  extent=(-180, 180, -90, 90),
                  alpha=0.5,
                  zorder=0)

        self.bbox.plot(ax=self.ax,
                       alpha=.1,
                       color='grey')
        self.graticule.plot(ax=self.ax,
                            color='lightgrey')
        # worldBound.plot(ax=ax,
        #                       alpha=0.5,
        #                       cmap='Greys')

        self.ax.set(title="World map",
               xlabel="X Coordinates (degrees)",
               ylabel="Y Coordinates (degrees)")

        for axis in [self.ax.xaxis, self.ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)  # For converting units to scientific ones (1ex).
            axis.set_major_formatter(formatter)

        plt.axis('equal')
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_map(self, locations):
        gpd = self.convert_points(locations)
        if self.points.empty:
            self.points = gpd
        else:
            self.points.update(gpd)
        self.points.plot(ax=self.ax, markersize=100, color='black')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # time.sleep(0.001)


    @staticmethod
    def analyze_biome_map():
        num_map = imageio.v3.imread(Map.BIOME_MAP_PATH)
        legend = Map.BIOME_LEGEND
        items = [item[1][0] for item in list(legend.items())]
        bins = np.array(items, dtype=int)
        processed_map = num_map
        biome_map = np.zeros(shape=(800, 1600), dtype=int)
        for i, line in enumerate(num_map):
            processed_map[i], biome_map[i] = Map.multidimensional_approximation(line, bins)
        return processed_map, biome_map

    @staticmethod
    def multidimensional_approximation(arr, bins):
        distances = np.sqrt(((arr - bins[:, np.newaxis, :]) ** 2).sum(axis=2))
        indexes = np.argmin(distances, axis=0)
        return [bins[i] for i in indexes], indexes

    @staticmethod
    def distance(point1, point2):
        return np.max([abs(point1[0] - point2[0]), abs(point1[1] - point2[1])])


if __name__ == "__main__":
    Map.BIOME_MAP_PATH = Map.BASE_PATH + r"\only_biome_map.png"

    # Create numpy array of x,y point locations
    add_points = np.array([[-5000, 0],
                           [0, 5000],
                           [1500, -4500]])
    mymap = Map()
    mymap.convert_points(add_points)
    mymap.plot_map()
