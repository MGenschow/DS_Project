#%% import packages
import pandas as pd
import numpy as np
import requests
import json
import zipfile
import glob
import os
import yaml
import re
import shutil

from random import randint
from time import sleep
from bs4 import BeautifulSoup
from datetime import datetime
#%% load configuration file
config_path = '/home/tu/tu_tu/tu_zxmny46/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% scraping class
class DWDScraper():

    def __init__(self):
        self.data_dir = config['data']['dwd']
        self.bounding_boxes = config['bboxes']['munich']
        self.url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/'

    def get_all_stations(self, min_datum, max_datum):

        os.chdir(self.data_dir + '/meta')

        r = requests.get(self.url + 'TU_Stundenwerte_Beschreibung_Stationen.txt')
        with open('TU_Stundenwerte_Beschreibung_Stationen.txt', 'wb') as f:
            f.write(r.content)
        stations = pd.read_fwf('TU_Stundenwerte_Beschreibung_Stationen.txt',
                               skiprows=2,
                               encoding = "ISO-8859-1",
                               header=None,
                               names = ['Stations_id','von_datum','bis_datum','Stationshoehe','geoBreite','geoLaenge','Stationsname','Bundesland'])

        stations['von_datum'] = pd.to_datetime(stations['von_datum'], format='%Y%m%d')
        stations['bis_datum'] = pd.to_datetime(stations['bis_datum'], format='%Y%m%d')
        stations = stations[(stations['von_datum'] <= min_datum) & (stations['bis_datum'] >= max_datum)]
        stations.columns = stations.columns.str.upper()
        os.remove('TU_Stundenwerte_Beschreibung_Stationen.txt')
        stations.to_csv(self.data_dir + '/meta/stations.csv', index=False)

    def get_relevant_station_ids(self, stations):
        foc = stations[(stations['GEOBREITE'] > self.bounding_boxes[1]) &
                       (stations['GEOBREITE'] < self.bounding_boxes[3]) &
                       (stations['GEOLAENGE'] > self.bounding_boxes[0]) &
                       (stations['GEOLAENGE'] < self.bounding_boxes[2])]
        return(foc.STATIONS_ID.tolist())
    
    def scrape(self, start_date, end_date, stations_id):

        os.chdir(self.data_dir + '/meta')

        start_date = datetime.strptime(start_date,"%Y-%m-%d")
        end_date = datetime.strptime(end_date,"%Y-%m-%d")
        dates = pd.date_range(start_date, end_date, freq='H')
        
        station_filter = pd.DataFrame({'MESS_DATUM': dates})
        station_temp = pd.DataFrame()
        cols = ['TT_TU', 'RF_TU']

        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]

        stations_id = [str(s).rjust(5,'0') for s in stations_id]
        j_stations_id = '|'.join(stations_id)
        reg = 'stundenwerte_TU_' + j_stations_id + '_[0-9]{8}_20221231_hist.zip'
        hrefs_final = list(filter(lambda x: re.findall(reg, x), hrefs))

        for i in range(len(hrefs_final)):
            filename = hrefs_final[i]
            r = requests.get(self.url + filename, stream=True)
            with open(filename, 'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(filename.replace('.zip',''))
            file = glob.glob(filename.replace('.zip','') + '/produkt_tu_stunde*')
            station_data = pd.read_csv(file[0], sep = ';')[["STATIONS_ID","MESS_DATUM","TT_TU","RF_TU"]]
            station_data['MESS_DATUM'] = pd.to_datetime(station_data['MESS_DATUM'], format='%Y%m%d%H')
            station_data = station_data.merge(station_filter, how="right", on='MESS_DATUM')
            station_temp = pd.concat([station_temp, station_data])
            shutil.rmtree(filename.replace('.zip',''))
            os.remove(filename)
            sleep(randint(1,5))

        station_temp[station_temp['RF_TU'] < 0]['RF_TU'] = np.nan
        station_temp[station_temp['TT_TU'] < -50]['TT_TU'] = np.nan

        station_temp.to_csv(self.data_dir + '/dwd.csv', index=False)