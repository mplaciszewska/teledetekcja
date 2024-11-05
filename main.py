from osgeo import gdal
gdal.UseExceptions()
from typing import Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import geopandas as gpd
import pandas as pd
import shapely
from rasterio.features import rasterize
import cv2


def read_spatial_raster(path: Union[str, Path]) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    assert dataset is not None, "Read spatial raster returned None"
    return dataset

raster_file = "CIR2015.tif"
raster_dataset = read_spatial_raster(raster_file)

def show_grayscale_matplotlib(array: np.ndarray):
    plt.imshow(array, cmap='gray')

red = raster_dataset.GetRasterBand(1)
red_array = red.ReadAsArray()
red_array = np.copy(red_array)

green = raster_dataset.GetRasterBand(2)
green_array = green.ReadAsArray()
green_array = np.copy(green_array)

blue = raster_dataset.GetRasterBand(3)
blue_array = blue.ReadAsArray()
blue_array = np.copy(blue_array)

rgb = np.dstack((red_array, green_array, blue_array))
plt.imshow(rgb, cmap='gray')
plt.show()

def points_to_pixels(points: np.ndarray, geotransform: List[float]) -> np.ndarray:
    c, a, _, f, _, e = geotransform
    columns = (points[:, 0] - c) / a
    rows = (points[:, 1] - f) / e
    pixels = np.vstack([rows, columns])
    pixels = pixels.T
    return pixels

def read_features_to_geopandas(path: Union[str, Path]) -> gpd.GeoDataFrame:
    features = gpd.read_file(path)
    return features


def reproject_geodataframe(features: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    return features.to_crs(crs)


def convert_to_pixel_system(features: gpd.GeoDataFrame, geotransform: List[float]) -> gpd.GeoDataFrame:
    def transform_function(xy: np.ndarray):
        ij = points_to_pixels(xy, geotransform)
        ji = ij[:, [1, 0]]
        return ji
    
    indices = features.index
    for i in indices:
        geometry = features.loc[i, "geometry"]
        geometry = shapely.transform(geometry, transform_function)  # To make our solution work for every type of geometry
        features.loc[i, "geometry"] = geometry
    return features
    
features_file = "drogi_4.shp"
features = read_features_to_geopandas(features_file)
features = reproject_geodataframe(features, raster_dataset.GetProjection())
features = convert_to_pixel_system(features, raster_dataset.GetGeoTransform())

example_feature = features.iloc[5]  # Select sample feature from our layer
example_polygon = example_feature["geometry"]

bounds = example_polygon.bounds
bounds = np.float64(bounds)
print("BBOX poligonu:", bounds)

bounds[:2] = np.floor(bounds[:2])
bounds[2:] = np.ceil(bounds[2:])
bounds = np.int64(bounds)
print("BBOX poligonu (integer):", bounds)

red_fragment = red_array[
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]
green_fragment = green_array[
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]
blue_fragment = blue_array[
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]

#wizualizacja przycietego do poligonu rastra
rgb_fragment = np.dstack((red_fragment, green_fragment, blue_fragment))
plt.imshow(rgb_fragment, cmap='gray')
plt.show()

# przesuniecie ukladu poligonu z duzego rastra na wyciety fragment
polygon_in_fragment_frame = shapely.affinity.translate(example_polygon, -bounds[0], -bounds[1])

def mask_fragment(fragment):
    no_data_mask = rasterize([polygon_in_fragment_frame], fragment.shape)
    no_data_mask = np.bool_(no_data_mask)
    no_data_mask = ~no_data_mask    # Rasterio puts True inside polygon
    masked_fragment = np.copy(fragment)
    masked_fragment[no_data_mask] = 0
    return masked_fragment

red_masked = mask_fragment(red_fragment)
green_masked = mask_fragment(green_fragment)
blue_masked = mask_fragment(blue_fragment)

#wizualizacja przycietego do poligonu rastra z wymaskowanymi pikselami
rgb_masked_fragment = np.dstack((red_masked, green_masked, blue_masked))
plt.imshow(rgb_masked_fragment, cmap='gray')
plt.show()

#obliczenie NDVI
NDVI = (red_array-green_array)/(red_array+green_array)
NDVI2 = (red_array+green_array)/(red_array-green_array)

fragmentNDVI = NDVI [
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]
fragmentNDVI2 = NDVI2 [
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]

IR = np.float32(red_array)
R = np.float32(green_array)
NDVI_float = (IR-R)/(IR+R)

fragmentNDVI_float = NDVI_float [
    bounds[1]: bounds[3],
    bounds[0]: bounds[2]
]
plt.imshow(fragmentNDVI_float, cmap = 'gray')

raster_as_float = fragmentNDVI_float
raster_norm = (raster_as_float - raster_as_float.min())/(raster_as_float.max() - raster_as_float.min())
raster_0_255 = raster_norm *255
raster_to_save = np.uint8(raster_0_255)
cv2.imwrite("raster.jpg", raster_to_save)
