#%% start-up
from HeatwaveM import HeatwaveM
import pandas as pd
import numpy as np
import pickle
import yaml
config_path = '/home/tu/tu_tu/tu_zxmny46/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% run
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')
dwd['MESS_DATUM'] = pd.to_datetime(dwd['MESS_DATUM'], format='%Y-%m-%d %H')
w = HeatwaveM(dwd)
w.get_heatwaves_ky()
w.heatwaves_summary()
w.save_heatwaves_to_list('heatwaves.pkl')
# %%
