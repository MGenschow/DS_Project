# %%
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from glob import glob
from tqdm import tqdm
import pickle
from shapely.geometry import Polygon, MultiPolygon

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
city_3d_model_path = config['data']['city_3d_model']

# %%
def print_element(element, indent=""):
    """
    Recursively prints the tag, attributes, and text content of an XML element and its child elements.

    Args:
        element (xml.etree.ElementTree.Element): The XML element to print.
        indent (str): The indentation string to use for nested elements.

    Returns:
        None
    """
    # Print element tag and attributes
    print(f"{indent}Tag: {element.tag}")
    for attribute, value in element.attrib.items():
        print(f"{indent}Attribute: {attribute} = {value}")
    
    # Print element text content, if available
    if element.text and element.text.strip():
        print(f"{indent}Text: {element.text.strip()}")

    # Recursively print child elements
    for child in element:
        print_element(child, indent + "  ")
#print_element(root)

# %%
def extract_fields(file_path):
    """
    Extracts specific fields from a CityGML file and returns a list of dictionaries representing the buildings.

    Args:
        file_path (str): The path to the CityGML file.

    Returns:
        list: A list of dictionaries representing the buildings. Each dictionary contains the extracted fields.

    """
    # Parse XML with ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()

    namespace = {'bldg': 'http://www.opengis.net/citygml/building/1.0',
                 'gen': 'http://www.opengis.net/citygml/generics/1.0',
                 'gml': 'http://www.opengis.net/gml'}

    buildings = []
    for building_elem in root.findall('.//bldg:Building', namespace):
        building = {}

        # ID of the building
        building['id'] = building_elem.get('{http://www.opengis.net/gml}id')
        # Elevation of the ground elements
        for attr_elem in building_elem.findall('.//gen:stringAttribute[@name="HoeheGrund"]/gen:value', namespace):
            building['HoeheGrund'] = float(attr_elem.text)
        # Elevation of the roof element
        for attr_elem in building_elem.findall('.//gen:stringAttribute[@name="HoeheDach"]/gen:value', namespace):
            building['HoeheDach'] = float(attr_elem.text)
        # Measured height of the building
        measuredHeight = building_elem.find('.//bldg:measuredHeight', namespace)
        if measuredHeight is not None:
            building['measuredHeight'] = float(measuredHeight.text)
        # Roof type
        roofType = building_elem.find('.//bldg:roofType', namespace)
        if roofType is not None:
            building['roofType'] = int(roofType.text)
        # building function
        function = building_elem.find('.//bldg:function', namespace)
        if function is not None:
            building['function'] = function.text
        # Initialize lists for RoofSurface_PosList and Dachneigung
        building['RoofSurface_PosList'] = []
        building['Dachneigung'] = []
        
        for roof_elem in building_elem.findall('.//bldg:RoofSurface', namespace):
            poslist = roof_elem.find('.//gml:posList', namespace)
            if poslist is not None:
                # Append poslist text to the list
                building['RoofSurface_PosList'].append(poslist.text)

            dachneigung = roof_elem.find('.//gen:stringAttribute[@name="Dachneigung"]/gen:value', namespace)
            if dachneigung is not None:
                # Append Dachneigung text to the list
                building['Dachneigung'].append(dachneigung.text)
           
        # ground shape
        for ground_elem in building_elem.findall('.//bldg:GroundSurface', namespace):
            poslist = ground_elem.find('.//gml:posList', namespace)
            if poslist is not None:
                building['GroundSurface_PosList'] = poslist.text

        buildings.append(building)
    
    return buildings



# %%
def convert_3d_coordinates_to_polygons(coordinates_list):
    """
    Converts a list of 3D coordinates into a MultiPolygon object.

    Args:
        coordinates_list (list): A list of 3D coordinates represented as strings.

    Returns:
        shapely.geometry.MultiPolygon: A MultiPolygon object representing the polygons.

    """
    polygons = []
    for coordinates in coordinates_list:
        # Split the string by whitespace and remove any leading/trailing whitespace
        coord_list = [c.strip() for c in coordinates.split()]

        # Extract only the x and y coordinates, ignoring the z coordinate
        xy_coords = [(float(coord_list[i]), float(coord_list[i+1])) for i in range(0, len(coord_list), 3)]

        # Create a shapely Polygon object from the xy coordinates
        polygon = Polygon(xy_coords)
        polygons.append(polygon)

    return MultiPolygon(polygons)

# %%
def process_all_gml_files(gml_path):
    """
    Processes all GML files in a given directory.

    Args:
        gml_path (str): The path to the directory containing the GML files.

    Returns:
        None

    This function performs the following steps to process the GML files:
    1. Retrieves a list of all GML files in the specified directory.
    2. Prints the number of GML files found.
    3. Iterates over each GML file and extracts the desired fields using the `extract_fields` function.
    4. Concatenates all the extracted dataframes into a single dataframe.
    5. Prints the number of objects in the final dataframe.
    6. Initiates the reprojecting and conversion to GeoPandas process.
    7. Converts the 'RoofSurface_PosList' coordinates in each row to polygons using the `convert_3d_coordinates_to_polygons` function.
    8. Creates a GeoDataFrame from the dataframe with the converted polygons, setting the 'geometry' column.
    9. Sets the coordinate reference system (CRS) of the GeoDataFrame to EPSG:25832.
    10. Converts the 'geometry' column to EPSG:4326 and stores it in the 'geometry_4326' column.
    11. Calculates a mask to identify gable roofs based on the 'Dachneigung' values.
    12. Saves the processed GeoDataFrame as a pickle file named 'processed_roofs.pkl' in the 'city_3d_model_path/processed' directory.
    """
    all_gml_files = glob(gml_path + '/*.gml')
    print(f'Processing {len(all_gml_files)} gml files')
    all_dfs = []
    for file in tqdm(all_gml_files):
        all_dfs.append(pd.DataFrame(extract_fields(file)))
    df = pd.concat(all_dfs)
    print(f'Processing finished, final dataframe has {len(df)} objects')
    print('\nStarting reprojecting and conversion to GeoPandas')
    df['geometry'] = df['RoofSurface_PosList'].apply(convert_3d_coordinates_to_polygons)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=25832, inplace=True)

    gdf['geometry_4326'] = gdf['geometry'].to_crs("EPSG:4326")
    # Get mask for flat and gable roofs
    gdf['Dachneigung'] = gdf['Dachneigung'].apply(lambda x: [float(value) for value in x])
    gdf['is_gable'] = gdf['Dachneigung'].apply(lambda x: any(round(value,2) != 90.0 for value in x))
    
    with open(city_3d_model_path + 'processed/processed_roofs.pkl', 'wb') as f:
        pickle.dump(gdf, f)

# %%
# Process all Files and save
process_all_gml_files(city_3d_model_path+'raw_gml')
