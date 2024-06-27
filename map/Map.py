import os

import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import Point


class Map:
    """
        Manages the graphical representation of the map, calculates distances, and divides the world into different biomes.

        Attributes:
            points (list): An updated list of points that can be displayed on the map when the visual display is activated.
            colored_biome_map (numpy.ndarray): A 3D array of the size of the map that holds the image of the map, divided into biomes. The third axis holds the RGB values.
            biome_map (numpy.ndarray): A 2D array that contains the biome division according to the numbers presented in the biome_legend.
            worldBound (shapely.geometry.Polygon): The definition of the shape of the world, holds a shapely object.
            graticule (shapely.geometry.LineString): Defines the lines drawn on the map, according to the angles of the world before its simplification to 2D.
            bbox (shapely.geometry.Polygon): The bounding box of the map. Represents the shape to which the 3D will be distorted to 2D. In this case, the bbox will be in the projection of wgs84, or plate caree, where the sizes of the continents are preserved in the correct proportions on the y-axis, but not on the x-axis.

        Constants:
            BASE_PATH (str): Contains the path to the file from which the program is run.
            BIOME_LEGEND (dict): A dictionary that interprets specific RGB colors as specific biomes.
            BIOME_MAP_PATH (str): The path to the map file. The map is colored so that each color represents a biome.
        """

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

    def __init__(self):
        """
            Initializes the Map object by importing necessary components from Shapely and setting up the map representation.
        """
        self.points = gpd.GeoDataFrame(columns=['geometry'],
                                       crs="ESRI:54001")
        self.colored_biome_map, self.biome_map = Map.analyze_biome_map()

        # Adjust plot font sizes
        sns.set(font_scale=1.5)
        sns.set_style("white")

        # Set working dir & get data
        # data = et.data.get_data('spatial-vector-lidar', verbose=True)

        # Import world boundary shapefile
        USER = r'DvirS'
        DRIVER = r'C:' + os.path.sep
        worldBound_path = os.path.join(DRIVER, 'Users', USER, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                       "ne_110m_land", "ne_110m_land.shp")
        self.worldBound = gpd.read_file(worldBound_path)

        # Import graticule & world bounding box shapefile data
        graticule_path = os.path.join(DRIVER, 'Users', USER, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                      "ne_110m_graticules_all", "ne_110m_graticules_15.shp")
        self.graticule = gpd.read_file(graticule_path)

        bbox_path = os.path.join(DRIVER, 'Users', USER, 'earth-analytics', "data", "spatial-vector-lidar", "global",
                                 "ne_110m_graticules_all", "ne_110m_wgs84_bounding_box.shp")
        self.bbox = gpd.read_file(bbox_path)

    def get_biome(self, coordinates):
        """
            Retrieves the biome type at a specified location on the map.

            Args:
                coordinates (Tuple[int, int]): A tuple representing the (x, y) coordinates of the location on the map.

            Returns:
                int: The biome ID corresponding to the biome type at the given coordinates.
        """
        return self.biome_map[tuple(np.flip(coordinates, 0))]

    @staticmethod
    def get_surroundings(matrix, coordinates, dtype=int):
        """
            Extracts a 3x3 neighborhood from a given map centered at the specified coordinates.

            Args:
                matrix (numpy.ndarray): The 2D array representing the map.
                coordinates (Tuple[int, int]): The (x, y) coordinates of the center point within the map.
                dtype (Optional[int]): The data type to use for padding values outside the map boundaries. Default is `int`.

            Returns:
                numpy.ndarray: A 3x3 subarray of the map centered at the given coordinates.
        """

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
        """
            Converts locations specified in distance coordinates (using Robinson projection) to WGS84 coordinates for map display.

            **Note:** This function is currently unused due to the complexity and performance implications of the Robinson projection.

            Args:
                locations (np.ndarray): A 2D NumPy array containing the locations in distance coordinates (x, y). The units are assumed to be kilometers.

            Returns:
                np.ndarray: A 2D NumPy array containing the converted locations in WGS84 coordinates (longitude, latitude).
        """

        # Turn points into list of x,y shapely points, and translate them from km to m.
        wgs84_locs = (locations - (800, 400)) / (800, -400) * (180, 90)
        points = [Point(xy) for xy in wgs84_locs]

        # Create geodataframe using the points
        gpds = gpd.GeoDataFrame(points,
                                columns=['geometry'],
                                crs="ESRI:54001")

        # Reproject point locations to the WGS84 projection
        # locations_WGS84 = locations.to_crs(self.worldBound.crs)

        # self.points = gpds
        return gpds

    def plot_map(self, locations=None, pops=None):
        """
            Generates and displays the graphical representation of the map.

            Args:
                :arg locations: This parameter contains the locations of the points needed to be plotted.
                :arg pops: This parameter contains the population numbers corresponding with the locations list.
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

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
        self.plot_points(locations, np.array(pops))
        plt.show()

    def plot_points(self, locations, pops):
        """
            Plots relevant points on the map, via the Map object's axes and figure.

            Args:
                locations (List): A list of locations to be displayed on the map.
                pops (ndarray): A list of population numbers corresponding with locations.
        """
        cmap = 'rainbow'
        min_cmap = 0
        max_cmap = np.max(pops)
        self.points = self.convert_points(locations)
        self.points.plot(ax=self.ax, markersize=100, column=pops, cmap=cmap, vmin=min_cmap, vmax=max_cmap)

        self.fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_cmap, max_cmap)), ax=self.ax,
                          orientation='vertical', format='%.0f', label="Population Density")

    @staticmethod
    def analyze_biome_map():
        """
        Analyzes the biome map and returns both a visualization-friendly representation and an informative data array.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - visualization_map: A 2D NumPy array representing the biome divisions, where each element corresponds to a biome ID.
                - biome_data: An array containing information about each biome, including its ID, name, and area.
        """
        num_map = imageio.v3.imread(Map.BIOME_MAP_PATH)
        legend = Map.BIOME_LEGEND
        items = [item[1][0] for item in list(legend.items())]
        bins = np.array(items, dtype=int)
        biome_map = Map.multidimensional_approximation(num_map, bins[:, np.newaxis, :])
        processed_map = bins[biome_map]
        return processed_map, biome_map

    @staticmethod
    def multidimensional_approximation(arr, bins):
        """
        Approximates a multidimensional array of values to a set of predefined bins.

        Args:
            arr (np.ndarray): The multidimensional array of values to be approximated.
            bins (np.ndarray): A tuple of arrays representing the binning boundaries for each dimension.

        Returns:
            np.ndarray: The approximated array, where each value has been mapped to the closest bin center.
        """
        distances = np.sqrt(((arr - bins[:, np.newaxis, :]) ** 2).sum(axis=len(arr.shape)))
        indexes = np.argmin(distances, axis=0)
        return indexes

    @staticmethod
    def distance(point1, point2):
        """
            Calculates the Manhattan distance (step distance) between two points on a coordinate grid.

            Args:
                point1 (Tuple[int, int]): The coordinates of the first point (x1, y1).
                point2 (Tuple[int, int]): The coordinates of the second point (x2, y2).

            Returns:
                int: The Manhattan distance (step distance) between the two points.
            """
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
