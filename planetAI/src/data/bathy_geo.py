import os
import itertools

import geopandas as gpd
from shapely import LineString, Polygon
from shapely.geometry import Point
# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
from rasterio.features import rasterize
from affine import Affine
import rasterio


from .utils import PlanetConfig, profile


def convert_to_web_mercator(lat_lon_array: np.ndarray) -> np.ndarray:
    lat_rad = lat_lon_array[:, 0] * np.pi / 180
    lon_rad = lat_lon_array[:, 1] * np.pi / 180
    r = 6378137
    x_mercator = r * lon_rad
    y_mercator = r * np.log(np.tan(np.pi / 4 + lat_rad / 2))
    return np.stack([y_mercator, x_mercator], axis=-1)


def distance_to_plate_boundary(lat_lon_array: np.ndarray, plate_boundaries: list[LineString]):
    """
    Given an array of (lat, lon) coordinates and a plate geometry,
    returns an array of distances (in meters) from each point to the plate boundary.

    Parameters:
        lat_lon_array (array-like): List or Nx2 array of (lat, lon)
        plate_geom (shapely Polygon or MultiPolygon): Plate geometry in (EPSG:3857)

    Returns:
        np.ndarray: Array of distances in meters
    """
    # Create GeoSeries of points in EPSG:4326
    mercator_array = convert_to_web_mercator(lat_lon_array)
    points = gpd.GeoSeries([Point(x, y) for y, x in mercator_array], crs="EPSG:3857")
    # print(points.head())

    # Compute distances
    min_distances = None
    for linestring in plate_boundaries:
        distances = points.distance(linestring)
        if min_distances is None:
            min_distances = distances
        # else:
        #     min_distances = np.minimum(min_distances, distances)

    return min_distances.values


def distance_to_plate(lat_lon_array: np.ndarray, plate_geom: Polygon):
    """
    Given an array of (lat, lon) coordinates and a plate geometry,
    returns an array of distances (in meters) from each point to the plate boundary.

    Parameters:
        lat_lon_array (array-like): List or Nx2 array of (lat, lon)
        plate_geom (shapely Polygon or MultiPolygon): Plate geometry in (EPSG:3857)

    Returns:
        np.ndarray: Array of distances in meters
    """
    # Create GeoSeries of points in EPSG:4326
    mercator_array = convert_to_web_mercator(lat_lon_array)
    points = gpd.GeoSeries([Point(x, y) for y, x in mercator_array], crs="EPSG:3857")
    # print(points.head())

    # Get boundary of the plate
    plate_boundary = plate_geom.boundary

    # Compute distances
    distances = points.distance(plate_boundary)

    return distances.values


def points_within_plate(plate_geom, resolution_meters=19000):
    """
    Given a plate geometry and resolution, return an array of (lat, lon) points inside the plate.

    Parameters:
        plate_geom (shapely Polygon or MultiPolygon): The plate geometry in geographic CRS (EPSG:4326)
        resolution_meters (int): Spacing between points in meters

    Returns:
        List of (latitude, longitude) tuples
    """

    # Get bounds
    minx, miny, maxx, maxy = plate_geom.bounds
    width = int((maxx - minx) / resolution_meters)
    height = int((maxy - miny) / resolution_meters)

    if width == 0 or height == 0:
        return []  # Geometry is too small for this resolution

    # Create affine transform
    transform = Affine.translation(minx, maxy) * Affine.scale(resolution_meters, -resolution_meters)

    # Rasterize polygon to a binary mask
    mask = rasterize(
        [(plate_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Get pixel indices where value is 1 (inside the polygon)
    rows, cols = np.nonzero(mask)

    # Convert to metric coordinates (x, y)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

    # Convert to geographic coordinates (lon, lat)
    points_metric = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:3857")
    points_geo = points_metric.to_crs("EPSG:4326")

    # Return as (lat, lon) tuples
    return [(pt.y, pt.x) for pt in points_geo]


@profile
def find_plate(gdf: gpd.GeoDataFrame, point: Point):
    # Use the spatial index to filter candidates first
    candidates = list(gdf.sindex.intersection(point.bounds))

    for idx in candidates:
        if gdf.iloc[idx].geometry.contains(point):
            return gdf.iloc[idx]  # return the full row, or gdf.iloc[idx]['PlateName'] etc.

    return None  # No containing polygon found


def get_distances(coord_tile: np.ndarray, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Given a tile of latitude and longitude coords with shape (w, h, 2),
    find the distance to the closest line
    """

    # 1. Get the bounds of the coord tile
    min_lon, min_lat = coord_tile.min(axis=(0, 1))
    max_lon, max_lat = coord_tile.max(axis=(0, 1))

    points = [
        Point(min_lon, min_lat),
        Point(max_lon, min_lat),
        Point(min_lon, max_lat),
        Point(max_lon, max_lat)
    ]
    for point in points:
        # Compute distance from the point to each line in the dataset
        gdf["distance_to_point"] = gdf.geometry.distance(point)

        # Sort by nearest
        gdf_sorted = gdf.sort_values("distance_to_point")

        # View the nearest lines
        print(gdf_sorted[["distance_to_point"]].head())


@profile
def main():
    # Load the shapefile
    gdf = gpd.read_file(os.path.join(PlanetConfig().data_dir, "PB2002_plates.shp"))

    print(gdf.head())
    #        LAYER Code   PlateName                                           geometry
    # 0  plate   AF      Africa  POLYGON ((-0.4379 -54.8518, -0.91466 -54.4535,...
    # 1  plate   AN  Antarctica  POLYGON ((180 -65.7494, 180 -90, -180 -90, -18...
    # 2  plate   SO     Somalia  POLYGON ((32.1258 -46.9998, 32.1252 -46.9975, ...
    # 3  plate   IN       India  POLYGON ((56.2652 14.6232, 57.0015 14.6601, 57...
    # 4  plate   AU   Australia  MULTIPOLYGON (((-180 -32.30415, -180 -15.62071...

    # Reproject to World Mercator (meters)

    h = 256
    tile_w = 256
    w = 2 * h
    rasterized = rasterize(
        [(geom, i) for i, geom in enumerate(gdf.geometry, 1)],
        out_shape=(h, w),  # Adjust resolution as needed
        transform=Affine.translation(-180, 90) * Affine.scale(360 / w, -180 / h),
        fill=0,
        dtype='int32'
    )

    # TODO Actually use plate boundary data for the distances

    boundaries_gdf = gpd.read_file(os.path.join(PlanetConfig().data_dir, "PB2002_boundaries.shp"))
    print(boundaries_gdf.head())

    gdf = gdf.to_crs("EPSG:3857")
    boundaries_gdf = boundaries_gdf.to_crs("EPSG:3857")

    print(gdf.head())
    print(boundaries_gdf.head())

    from PIL import Image

    # random_start_y = 0
    # random_start_x = 0

    # tile = rasterized.copy()

    distances = np.zeros((h, w), dtype=np.float32)
    # random_start_y = np.random.randint(0, h - tile_w)
    # random_start_x = np.random.randint(0, w - tile_w)
    for y, x in tqdm(list(itertools.product(range(0, h, tile_w), range(0, w, tile_w)))):
        tile = rasterized[y:y + tile_w, x:x + tile_w]

        for plate_num in np.unique(tile):
            if plate_num == 0:
                continue
            ys, xs = np.where(tile == plate_num)
            ys += y
            xs += x
            # Convert to lat/lon coordinates
            lat_lon_coords = np.array((90 - ((ys) * 180 / h), -180 + ((xs) * 360 / w)))
            # Get distances to the plate boundary
            plate_code = gdf[gdf.index == plate_num - 1]["Code"].values[0]
            boundaries = boundaries_gdf[
                (boundaries_gdf["PlateA"] == plate_code) | (boundaries_gdf["PlateB"] == plate_code)
            ]["geometry"].tolist()
            plate_distances = distance_to_plate_boundary(lat_lon_coords.T, boundaries)
            distances[ys, xs] = plate_distances

    distances = cv2.normalize(distances, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Image.fromarray(distances).save(os.path.join(PlanetConfig().test_dir, "bathy_distance.png"))
    return


if __name__ == "__main__":
    main()
