import os
import rasterio
from rasterio.warp import reproject, Resampling
import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import earthpy as et
import cartopy.crs as ccrs
from scipy.ndimage import map_coordinates
from PIL import Image


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
    BIOME_MAP_PATH = BASE_PATH + r"\map\only biome map.tiff"
    ADJUSTED_MAP_PATH = BASE_PATH + r'\map\adjusted map.tiff'

    def __init__(self, biome_map):
        # Adjust plot font sizes
        sns.set(font_scale=1.5)
        sns.set_style("white")

        # Set working dir & get data
        # data = et.data.get_data('spatial-vector-lidar', verbose=True)
        os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

        # Import world boundary shapefile
        worldBound_path = os.path.join("data", "spatial-vector-lidar", "global",
                                       "ne_110m_land", "ne_110m_land.shp")
        worldBound = gpd.read_file(worldBound_path)

        cmap = mpl.colors.ListedColormap([Map.BIOME_LEGEND[i][0] for i in range(11)])
        norm = mpl.colors.BoundaryNorm(list(range(11)), cmap.N)

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

        robinson_proj = ccrs.Robinson()
        rows, cols = biome_map.shape
        y, x = np.mgrid[0:rows, 0:cols]
        trans_points = robinson_proj.transform_points(ccrs.PlateCarree(), x, y)
        lon, lat, *extra_values = trans_points
        warped_biome_map = map_coordinates(biome_map, [lat, lon], order=1)  # Order 1 for linear interpolation

        # Reproject point locations to the Robinson projection
        city_locations_robin = city_locations.to_crs(worldBound_robin.crs)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': robinson_proj})

        ax.imshow(warped_biome_map,
                  origin="upper",
                  norm=norm,
                  cmap=cmap,
                  extent=(-20037500, 20037500, -10007555, 10007555),
                  alpha=0.5,
                  zorder=0)

        bbox_robin.plot(ax=ax,
                           alpha=.1,
                           color='grey')
        graticule_robin.plot(ax=ax,
                                color='lightgrey')
        worldBound_robin.plot(ax=ax,
                              alpha=0.5,
                              cmap='Greys')

        ax.set(title="World map (robinson)",
               xlabel="X Coordinates (meters)",
               ylabel="Y Coordinates (meters)")

        # city_locations_robin.plot(ax=ax, markersize=100, color='springgreen')

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

    @staticmethod
    def reproject_image(input_image_path, output_image_path, dst_crs="ESRI:54030"):
        """
        Reprojects an image (PNG or GeoTIFF) to Robinson projection.

        Args:
            input_image_path: Path to the input image (PNG or GeoTIFF).
            output_image_path: Path to the output image (GeoTIFF).
            dst_crs: CRS code for the target projection (default: Robinson).
        """
        # Open the input image
        r_src = rasterio.open(input_image_path, "r")
        bounds = r_src.bounds

        if r_src.crs is None:
            bounds = r_src.bounds
            r_src.close()
            w_src = rasterio.open(input_image_path, "r+", driver="GTiff")
            # Add basic geospatial info if missing
            w_src.crs = "epsg:4326"  # Assuming plate carrée
            w_src.transform = rasterio.Affine.from_gdal(*bounds)
            w_src.close()
            src = rasterio.open(input_image_path, "r")
        else:
            src = rasterio.open(input_image_path, "r")

        # Reproject the image
        with rasterio.open(
                output_image_path,
                "w",
                driver="GTiff",
                height=src.height,
                width=src.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=dst_crs,
                transform=reproject(
                    src_crs=src.crs, dst_crs=dst_crs, source=src # , src_width=src.width, src_hight=src.height, *src.bounds
                )
        ) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=dst.crs,
                    resampling=Resampling.nearest,
                )

        print(f"Image reprojected successfully to {dst_crs}!")


if __name__ == "__main__":
    cbm, bm = Map.analyze_biome_map(Map.BIOME_MAP_PATH)  # cbm: Colored Biome Map, bm: Biom Map
    Map.reproject_image(Map.BIOME_MAP_PATH, Map.ADJUSTED_MAP_PATH)
    # mymap = Map(bm)
