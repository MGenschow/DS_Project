#%% import packages
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import folium
import folium.raster_layers
import folium.features
import rasterio
import rioxarray as rxr

from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon
from shapely.geometry import box
from branca.colormap import LinearColormap
#%% haversine
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the Earth using the Haversine formula.

    Args:
        lon1 (float): Longitude of the first point in decimal degrees.
        lat1 (float): Latitude of the first point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.

    Returns:
        float: The distance between the two points in kilometers.

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#%% convert_lon_lat_to_lat_lon
def convert_lon_lat_to_lat_lon(point):
    """
    Convert a point in longitude and latitude format to latitude and longitude format.

    Args:
        point (list): A list containing the longitude and latitude coordinates in decimal degrees.

    Returns:
        list: A list containing the latitude and longitude coordinates in decimal degrees.

    """
    return [point[1], point[0]]

#%%
def convert_bbox_lon_lat_to_lat_lon(bbox):
    """
    Convert a bounding box in longitude and latitude format to latitude and longitude format.

    Args:
        bbox (list): A list containing the bounding box coordinates in the order [lon1, lat1, lon2, lat2].

    Returns:
        list: A list containing the converted bounding box coordinates in the order [lat1, lon1, lat2, lon2].

    """
    # Convert the first two coordinates (lon1, lat1)
    converted_bbox = [convert_lon_lat_to_lat_lon(bbox[:2])]

    # Convert the last two coordinates (lon2, lat2)
    converted_bbox.append(convert_lon_lat_to_lat_lon(bbox[2:]))

    # Flatten the converted_bbox list
    return [coord for point in converted_bbox for coord in point]

#%% create_polygon_from_coord
def create_polygon_from_coord(coordinates):
    """
    Create a polygon from the given coordinates.

    Args:
        coordinates (list): A list of four coordinates representing the bounding box of the polygon.
                            The order of the coordinates should be [lon1, lat1, lon2, lat2],
                            where lon1 represents the longitude of the lower-left corner,
                            lat1 represents the latitude of the lower-left corner,
                            lon2 represents the longitude of the upper-right corner,
                            and lat2 represents the latitude of the upper-right corner.

    Returns:
        GeoDataFrame: A GeoDataFrame containing a single polygon geometry representing the bounding box.
                      The GeoDataFrame has the 'EPSG:4326' coordinate reference system (CRS).

    """
    # Define the vertices of the polygon using the given coordinates
    vertices = [
        (coordinates[0], coordinates[3]),
        (coordinates[2], coordinates[3]),
        (coordinates[2], coordinates[1]),
        (coordinates[0], coordinates[1])
    ]

    # Create a Polygon geometry from the vertices
    polygon = Polygon(vertices)

    # Create a GeoDataFrame with the polygon geometry
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])

    # Set the coordinate reference system (CRS) of the GeoDataFrame
    polygon_gdf.crs = 'EPSG:4326'

    return polygon_gdf

#%% divide_polygon_into_grid
def divide_polygon_into_grid(polygon, grid_size_meters):
    """
    Divide a polygon into a grid of smaller polygons.

    Args:
        polygon (Polygon): The input polygon to be divided.
        grid_size_meters (float): The size of each grid cell in meters.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the grid polygons.

    """
    # Calculate the longitudinal and latitudinal distances of the polygon
    lon_distance = polygon.bounds[2] - polygon.bounds[0]
    lat_distance = polygon.bounds[3] - polygon.bounds[1]
    
    # Calculate the grid size in terms of longitudinal and latitudinal units
    lon_grid_size = grid_size_meters / (111320 * math.cos(math.radians(polygon.bounds[1])))
    lat_grid_size = grid_size_meters / 111320
    
    # Initialize a list to store the grid polygons
    grid_polygons = []
    
    # Iterate over the range of longitudinal and latitudinal values to create grid polygons
    for lon in np.arange(polygon.bounds[0], polygon.bounds[2], lon_grid_size):
        for lat in np.arange(polygon.bounds[1], polygon.bounds[3], lat_grid_size):
            # Create a square polygon using the current coordinates and grid sizes
            square = Polygon([
                (lon, lat), (lon + lon_grid_size, lat),
                (lon + lon_grid_size, lat + lat_grid_size), (lon, lat + lat_grid_size)
            ])
            
            # Check if the polygon intersects with the current square
            if polygon.intersects(square):
                intersection = polygon.intersection(square)
                
                # Check if the intersection area is greater than zero
                if intersection.area > 0:
                    # Handle different intersection geometry types
                    if intersection.geom_type == 'Polygon':
                        if intersection.equals(square):
                            grid_polygons.append(intersection)
                    elif intersection.geom_type == 'MultiPolygon':
                        for part in intersection:
                            if part.equals(square):
                                grid_polygons.append(part)
    
    # Create a GeoDataFrame from the grid polygons
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons)
    grid_gdf.crs = 'EPSG:4326'

    # Create IDs
    grid_gdf['id'] = 1000000 + np.arange(1,len(grid_gdf)+1)
    
    return grid_gdf

#%% calculate_surface_coverage
def calculate_surface_coverage(grid, inp, surface_labels):
    """
    Calculate the surface coverage fractions for each square in a grid.

    Args:
        grid (GeoDataFrame): A GeoDataFrame representing the grid squares.
        inp (GeoDataFrame): A GeoDataFrame representing the surface polygons.
        surface_labels (list): A list of surface labels to consider.

    Returns:
        DataFrame: A pandas DataFrame containing the results with the following columns:
                   - 'id': The identifier of the square.
                   - 'surface_fractions': A dictionary containing the surface label as key and
                                         the corresponding fraction as value.

    """
    # Create an empty list to store the results
    results = []

    # Iterate over each square in the grid dataframe
    for idx, square in grid.iterrows():
        # Get the geometry of the square
        square_geom = square.geometry

        # Create an empty dictionary to store surface areas
        surface_areas = {label: 0 for label in surface_labels}

        # Iterate over each surface polygon in the inp dataframe
        for inp_idx, surface_polygon in inp.iterrows():
            # Get the intersection between the square and surface polygon
            intersection = square_geom.intersection(surface_polygon.geometry)

            # Check if the intersection is valid and non-empty
            if not intersection.is_empty and intersection.area > 0:
                # Calculate the area of the intersection
                intersection_area = intersection.area

                # Get the surface label for the current polygon
                surface_label = surface_polygon['label']

                # Update the surface area dictionary
                surface_areas[surface_label] += intersection_area

        # Calculate the total area of the square
        square_area = square_geom.area

        # Calculate the fraction of each surface type in the square
        surface_fractions = {surface_label: area / square_area for surface_label, area in surface_areas.items()}

        # Append the results for the current square to the list
        results.append({'id': square['id'], 'surface_fractions': surface_fractions})

    # Convert the list of results to a pandas DataFrame
    result_df = pd.DataFrame(results)

    return result_df

#%% calculate_surface_coverage_fast
def calculate_surface_coverage_fast(grid, inp):
    """
    Calculate the surface coverage fractions for each square in a grid.

    Args:
        grid (GeoDataFrame): A GeoDataFrame representing the grid squares.
        inp (GeoDataFrame): A GeoDataFrame representing the surface polygons.

    Returns:
        DataFrame: A pandas DataFrame containing the results with the following columns:
                   - 'id': The identifier of the square.
                   - 'surface_fractions': A dictionary containing the surface label as key and
                                         the corresponding fraction as value.

    """
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each square in the grid GeoDataFrame
    for idx, square in grid.iterrows():
        
        # Get the geometry of the square
        square_geom = square.geometry
        
        # Create a subset GeoDataFrame with the square geometry
        subset = gpd.GeoDataFrame(geometry=[square_geom])
        
        # Perform intersection between the surface polygons and the subset
        intersection = gpd.overlay(inp, subset, how='intersection')

        # Calculate the total area of the total square
        total_area = square_geom.area

        # Calculate the area of the labels
        intersection['area'] = intersection.geometry.area

        # Calculate the area for each surface label and divide by the total area to get fractions
        grouped = (intersection.groupby('label')['area'].sum() / total_area).reset_index()
        
        # Transpose the DataFrame and set the surface labels as the index
        grouped = grouped.set_index('label').T.reset_index(drop=True)
        
        # Add the 'id' column with the identifier of the square
        grouped['id'] = square['id']

        # Append the results for the current square to the DataFrame
        result_df = pd.concat([result_df, grouped])

    # Reset the index and column names
    result_df = result_df.reset_index(drop=True)
    result_df.index.name = None
    
    # Reorder the columns
    result_df = result_df[['id', 'building', 'impervious', 'low vegetation', 'road', 'trees', 'water']]
    
    # Fill missing values with zero
    result_df.fillna(0, inplace=True)

    return result_df

#%% calculate_surface_coverage_super_fast
def calculate_surface_coverage_super_fast(grid, inp, surface_labels):
    """
    Calculate the surface coverage fractions for each square in a grid using a super-fast approach.

    Args:
        grid (GeoDataFrame): A GeoDataFrame representing the grid squares.
        inp (GeoDataFrame): A GeoDataFrame representing the surface polygons.
        surface_labels (list): A list of surface labels.

    Returns:
        DataFrame: A pandas DataFrame containing the results with the following columns:
                   - 'id': The identifier of the square.
                   - 'surface_fractions': A dictionary containing the surface label as key and
                                         the corresponding fraction as value.

    """
    # Create an empty list to store the results
    results = []

    # Iterate over each square in the grid dataframe
    for idx, square in grid.iterrows():
        # Get the geometry of the square
        square_geom = square.geometry

        # Create a bounding box from the square's coordinates
        bbox = box(*rec_polygon_coords(square_geom))

        # Select the surface polygons that intersect with the bounding box
        sub = inp[inp.geometry.intersects(bbox)]

        # Create an empty dictionary to store surface areas
        surface_areas = {label: 0 for label in surface_labels}

        # Iterate over each surface polygon in the inp dataframe
        for sub_idx, surface_polygon in sub.iterrows():
            # Get the intersection between the square and surface polygon
            intersection = square_geom.intersection(surface_polygon.geometry)

            # Check if the intersection is valid and non-empty
            if not intersection.is_empty and intersection.area > 0:
                # Calculate the area of the intersection
                intersection_area = intersection.area

                # Get the surface label for the current polygon
                surface_label = surface_polygon['label']

                # Update the surface area dictionary
                surface_areas[surface_label] += intersection_area

        # Calculate the total area of the square
        square_area = square_geom.area

        # Calculate the fraction of each surface type in the square
        surface_fractions = {surface_label: area / square_area for surface_label, area in surface_areas.items()}

        # Append the results for the current square to the list
        results.append({'id': square['id'], 'surface_fractions': surface_fractions})

    # Convert the list of results to a pandas DataFrame
    result_df = pd.DataFrame(results)

    return result_df

#%% calculate_average_height
def calculate_average_height(grid, wind):
    """
    Calculate the average height within each square of a grid based on the intersection with wind polygons.

    Args:
        grid (GeoDataFrame): A GeoDataFrame representing the grid with square polygons.
        wind (GeoDataFrame): A GeoDataFrame representing the wind polygons with measured heights.

    Returns:
        DataFrame: A pandas DataFrame with the calculated average heights for each square in the grid.

    """
    # Create an empty list to store the results
    results = []

    # Iterate over each square in the grid GeoDataFrame
    for idx, square in grid.iterrows():
        # Get the geometry of the square
        square_geom = square.geometry
        a = square_geom.area

        # Initialize the average height to zero for each square
        avg_height = 0

        # Iterate over each wind polygon in the wind GeoDataFrame
        for w_idx, w_polygon in wind.iterrows():
            # Get the intersection between the square and wind polygon
            intersection = square_geom.intersection(w_polygon.geometry_4326)

            # Check if the intersection is valid and non-empty
            if not intersection.is_empty and intersection.area > 0:
                # Add the weighted height value of the wind polygon to the average height of the square
                avg_height += (intersection.area / a) * w_polygon.measuredHeight

        # Append the average height for the current square to the results list
        results.append(avg_height)

    # Convert the list of results to a pandas DataFrame
    result_df = pd.DataFrame({'id': grid['id'], 'avg_height': results})

    return result_df

#%% calculate_average_height_super_fast
def calculate_average_height_super_fast(grid, wind):
    """
    Calculate the average height within each square of a grid based on the intersection with wind polygons using a quicker approach.

    Args:
        grid (GeoDataFrame): A GeoDataFrame representing the grid with square polygons.
        wind (GeoDataFrame): A GeoDataFrame representing the wind polygons with measured heights.

    Returns:
        DataFrame: A pandas DataFrame with the calculated average heights for each square in the grid.

    """
    # Create an empty list to store the results
    results = []

    # Iterate over each square in the grid GeoDataFrame
    for idx, square in grid.iterrows():
        # Get the geometry of the square
        square_geom = square.geometry

        # Create a bounding box from the square's coordinates
        bbox = box(*rec_polygon_coords(square_geom))

        # Select the surface polygons that intersect with the bounding box
        sub = wind[wind.geometry_4326.intersects(bbox)]

        # Get the total area of the square
        a = square_geom.area

        # Initialize the average height to zero for each square
        avg_height = 0

        # Iterate over each polygon in the subset GeoDataFrame
        for s_idx, w_polygon in sub.iterrows():
            # Get the intersection between the square and subset polygon
            intersection = square_geom.intersection(w_polygon.geometry_4326)

            # Check if the intersection is valid and non-empty
            if not intersection.is_empty and intersection.area > 0:
                # Add the weighted height value of the wind polygon to the average height of the square
                avg_height += (intersection.area / a) * w_polygon.measuredHeight

        # Append the average height for the current square to the results list
        results.append(avg_height)

    # Convert the list of results to a pandas DataFrame
    result_df = pd.DataFrame({'id': grid['id'], 'avg_height': results})

    return result_df

#%% convert_to_surface_dataframe
def convert_dict_to_cols(df):
    """
    Convert a DataFrame column containing dictionaries to separate columns.

    Args:
        df (DataFrame): A pandas DataFrame containing a column with dictionaries.

    Returns:
        DataFrame: A new DataFrame with the dictionaries converted to separate columns.

    """
    def extract_values(row):
        """
        Extract the values from a dictionary in a DataFrame row.

        Args:
            row (Series): A row from a pandas DataFrame.

        Returns:
            Series: A pandas Series containing the values extracted from the dictionary.

        """
        return pd.Series(row['surface_fractions'])

    # Apply the extract_values function to each row of the DataFrame
    new_df = df.apply(extract_values, axis=1)

    # Concatenate the original DataFrame with the new DataFrame containing the extracted values
    concat_df = pd.concat([df, new_df], axis=1)

    return concat_df

#%% pixels_to_foliumMap
def pixels_to_foliumMap(array, polygon, crs='EPSG4326'):
    """
    Convert pixel array data to a Folium map with overlays.

    Args:
        array (ndarray): The pixel array data.
        polygon (Polygon): The polygon representing the bounds of the pixel array.

    Returns:
        folium.Map: A Folium map with overlays.

    """
    # Extract the pixel array data
    data = np.array(array)[0]

    # Define color range for colormap
    color_range = np.linspace(0, 1, 256)
    colors_jet_rgba = plt.cm.jet(color_range)
    colors_jet_hex = [mcolors.rgb2hex(color) for color in colors_jet_rgba]

    # Create colormap and normalize the data
    cmap = plt.colormaps['jet']
    norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
    colored_data = cmap(norm(data))

    # Calculate image bounds and corner coordinates
    image_bounds = box(*array.rio.bounds())
    min_x, min_y, max_x, max_y = array.rio.bounds()
    corner_coordinates = [[min_y, min_x], [max_y, max_x]]

    # Create a Folium map centered on the image bounds
    m = folium.Map(
        location=[image_bounds.centroid.y, image_bounds.centroid.x],
        zoom_start=15,
        crs=crs
    )

    # Add a GeoJson overlay for the image bounds
    folium.GeoJson(
        image_bounds.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'orange', 'colorOpacity': 0.7}
    ).add_to(m)

    # Add a tile layer to the map
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='CartoDB',
        transparent=True,
    ).add_to(m)

    # Add the colored data as an ImageOverlay to the map
    folium.raster_layers.ImageOverlay(
        colored_data,
        bounds=corner_coordinates,
        opacity=0.5,
        interactive=True,
        cross_origin=True,
        pixelated=True,
        zindex=0.2
    ).add_to(m)

    # Create a colormap for the data and add it to the map
    colormap = LinearColormap(
        colors=colors_jet_hex,
        vmin=data.min(),
        vmax=data.max(),
        max_labels=15
    )
    colormap.caption = 'Land surface temperature in celsius'
    colormap.add_to(m)

    # Add a GeoJson overlay for the polygon
    folium.GeoJson(
        polygon.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'black', 'colorOpacity': 0.7}
    ).add_to(m)

    return m

#%% rec_polygon_coords
def rec_polygon_coords(polygon):
    """
    Retrieve the corner coordinates of a rectangle polygon.

    Args:
        polygon (Polygon): The rectangle polygon.

    Returns:
        tuple: A tuple containing the minimum x-coordinate, minimum y-coordinate,
               maximum x-coordinate, and maximum y-coordinate of the rectangle.

    """
    # Retrieve the points of the polygon's exterior
    points = list(polygon.exterior.coords)

    # Separate the x and y coordinates
    x_coordinates, y_coordinates = zip(*points)

    # Return the corner coordinates of the rectangle
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)

#%% pixel mean functions
def naive_pixel_mean(array, polygon):
    """
    Calculate the mean pixel value within a given polygon using a naive approach.

    Args:
        array (Rasterio DatasetReader): The raster array.
        polygon (Polygon): The polygon representing the area of interest.

    Returns:
        float: The mean pixel value within the polygon.

    """
    # Clip the raster array to the polygon extent
    c = array.rio.clip([polygon], crs=4326, all_touched=True)
    
    # Calculate the mean pixel value
    m = c.mean().values

    return m.item()


def naive_pixel_mean_wrapper(row, array):
    """
    Wrapper function to calculate the mean pixel value within a polygon for a row in a DataFrame.

    Args:
        row (Series): A row in a DataFrame containing a 'geometry' column representing the polygon.
        array (Rasterio DatasetReader): The raster array.

    Returns:
        float: The mean pixel value within the polygon.

    """
    # Extract the polygon from the row
    polygon = row.geometry
    
    # Calculate the mean pixel value using the naive_pixel_mean function
    result = naive_pixel_mean(array, polygon)

    return result


def weighted_pixel_mean(array, polygon, a=70):
    """
    Calculate the weighted mean pixel value within a given polygon.

    Args:
        array (Rasterio DatasetReader): The raster array.
        polygon (Polygon): The polygon representing the area of interest.
        a (float, optional): The weighting factor. Default is 70.

    Returns:
        tuple: A tuple containing the weights array and the weighted mean pixel value.

    """
    # Clip the raster array to the polygon extent
    c = array.rio.clip([polygon], crs=4326, all_touched=True)
    
    # Calculate the bounds of the clipped array
    minx, miny, maxx, maxy = c.rio.bounds()
    
    # Calculate the coordinates of the rectangle polygon
    p_minx, p_miny, p_maxx, p_maxy = rec_polygon_coords(polygon)
    
    # Calculate the weights based on the distances between the polygon and the clipped array bounds
    bottom = a - haversine(p_minx, p_miny, p_minx, miny) * 1000
    right = a - haversine(p_maxx, p_miny, maxx, p_miny) * 1000
    top = a - haversine(p_maxx, p_maxy, p_maxx, maxy) * 1000
    left = a - haversine(p_minx, p_maxy, minx, p_maxy) * 1000
    
    # Retrieve the pixel values from the clipped array
    npy = c[0].values
    npy[np.isnan(npy)] = np.nanmean(npy)
    
    # Create a weights array with ones of the same shape as npy
    weights = np.ones_like(npy)
    
    # Calculate the indices of the last row and column in the weights array
    ro = weights.shape[0] - 1
    co = weights.shape[1] - 1
    
    # Assign the weights based on the distances to the appropriate edges
    weights[:, 0] = left / a
    weights[:, co] = right / a
    weights[0, :] = top / a
    weights[ro, :] = bottom / a
    weights[0, 0] = (left / a) * (top / a)
    weights[0, co] = (right / a) * (top / a)
    weights[ro, 0] = (left / a) * (bottom / a)
    weights[ro, co] = (right / a) * (bottom / a)
    
    # Cap the weights at 1
    weights[weights > 1] = 1
    
    # Calculate the weighted mean pixel value
    m = np.average(npy, axis=None, weights=weights)

    return weights, m


def weighted_pixel_mean_wrapper(row, array):
    """
    Wrapper function to calculate the weighted mean pixel value within a polygon for a row in a DataFrame.

    Args:
        row (Series): A row in a DataFrame containing a 'geometry' column representing the polygon.
        array (Rasterio DatasetReader): The raster array.

    Returns:
        float: The weighted mean pixel value within the polygon.

    """
    # Extract the polygon from the row
    polygon = row.geometry
    
    # Calculate the weighted mean pixel value using the weighted_pixel_mean function
    _, result = weighted_pixel_mean(array, polygon)
    
    return result