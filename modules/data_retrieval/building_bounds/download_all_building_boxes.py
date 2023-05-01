# %%
import preprocess_building_boxes
from shapely.geometry import box

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config

# %%
bbox = box(*config['bboxes']['munich'])
building_box_downloader = preprocess_building_boxes.BuildingBoxDownloader(bbox)
building_box_downloader.prepare_relevant_building_boxes()
# %%
