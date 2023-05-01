# %%
import yaml
import preprocessing_orthophotos
from shapely.geometry import box

import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config


# %%
# Create index for othophotos, if not already exist
indexer = preprocessing_orthophotos.OrthophotoIndexer()
indexer.create_complete_index()

# %%
# Print summary of orthophoto download that is necessary
bbox = box(*config['bboxes']['munich'])
loader = preprocessing_orthophotos.OrthophotoLoader(data_dir = config['data']['orthophotos'], bbox=bbox, download=True)
loader.print_report()

# %%
loader.download_missing_tiles()

# %%
