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
#%% set directory
home_directory = os.path.expanduser( '~' )
os.chdir(home_directory + '/DS_Project/modules')
#%% load configuration file
config_path = 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% scraping class
class DWDScraper():
    """A class for scraping weather data from DWD.

    Attributes:
        data_dir (str): The directory path where the data will be stored.
        bounding_boxes (list): The bounding box coordinates for filtering stations.
        url (str): The base URL for accessing the DWD weather data.

    Methods:
        get_all_stations(min_datum, max_datum):
            Retrieves information about all weather stations.

        get_relevant_station_ids(stations):
            Filters and returns relevant station IDs based on bounding boxes.

        scrape(start_date, end_date, stations_id):
            Scrapes weather data for the specified date range and station IDs.
    """

    def __init__(self):
        """Initializes the DWDScraper object with default values."""
        self.data_dir = config['data']['dwd']
        self.bounding_boxes = config['bboxes']['munich']
        self.url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/'

    def get_all_stations(self, name, min_datum, max_datum):
        """Retrieves information about all weather stations.

        Args:
            name (str): Desired name of the written csv file.
            min_datum (datetime): The minimum date for filtering stations.
            max_datum (datetime): The maximum date for filtering stations.

        Returns:
            None
        """

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
        stations.to_csv(self.data_dir + '/meta/' + name, index=False)

    def get_relevant_station_ids(self, stations):
        """Filters and returns relevant station IDs based on bounding boxes.

        Args:
            stations (DataFrame): The DataFrame containing station information.

        Returns:
            list: A list of relevant station IDs.
        """

        foc = stations[(stations['GEOBREITE'] > self.bounding_boxes[1]) &
                       (stations['GEOBREITE'] < self.bounding_boxes[3]) &
                       (stations['GEOLAENGE'] > self.bounding_boxes[0]) &
                       (stations['GEOLAENGE'] < self.bounding_boxes[2])]
        return foc.STATIONS_ID.tolist()

    def scrape(self, name, start_date, end_date, stations_id):
        """Scrapes weather data for the specified date range and station IDs.

        Args:
            name (str): Desired name of the written csv file.
            start_date (str): The start date of the data collection (format: 'YYYY-MM-DD').
            end_date (str): The end date of the data collection (format: 'YYYY-MM-DD').
            stations_id (list): A list of station IDs to scrape data from.

        Returns:
            None
        """

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

        station_temp.to_csv(self.data_dir + '/' + name, index=False)