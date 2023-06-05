#%% import packages
import pandas as pd
import numpy as np
import pickle
import yaml
config_path = '/home/tu/tu_tu/tu_zxmny46/DS_Project/modules/config.yml'
# config_path = 'C:/Users/stefan/OneDrive - bwedu/04_semester/DS_Project/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% standard functions
def apparent_temperature_nws(t,h):

    """
    Calculate the apparent temperature using the NWS formula.

    Args:
        t (float): The air temperature in degrees Celsius.
        h (float): The relative humidity in percentage.

    Returns:
        float: The calculated apparent temperature.

    """

    if t <= 40:
        HIF = t
    else:
        A = -10.3 + 1.1*t + 0.047*h
        if A < 79:
            HIF = A
        else:
            B = (-42.379) + ((2.04901523)*t) + ((10.14333127)*h) + ((-0.22475541)*t*h) + ((-6.83783*1e-3)*((t)**2)) + ((-5.481717*1e-2)*((h)**2)) + ((1.22874*1e-3)*((t)**2)*h) + ((8.5282*1e-4)*t*((h)**2)) + ((-1.99*1e-6)*((t)**2)*((h)**2))
            if (h <= 13) & (t >= 80) & (t <= 112):
                HIF = B - ((13- h)/4)*(((17 - abs(t - 95))/17)**0.5)
            elif (h > 85) & (t >= 80) & (t <= 87):
                HIF = B + 0.02*(h - 85)*(87 - t)
            else:
                HIF = B
    return HIF

def apparent_temperature_std(t,h):

    """
    Calculate the apparent temperature using the Steadman formula.

    Args:
        t (float): The air temperature in degrees Celsius.
        h (float): The relative humidity in percentage.

    Returns:
        float: The calculated apparent temperature.

    """

    if t <= 80:
        HIF = t
    elif h <= 40:
        HIF = t
    else:
        HIF = (-42.379) + ((2.04901523)*t) + ((10.14333127)*h) + ((-0.22475541)*t*h) + ((-6.83783*1e-3)*((t)**2)) + ((-5.481717*1e-2)*((h)**2)) + ((1.22874*1e-3)*((t)**2)*h) + ((8.5282*1e-4)*t*((h)**2)) + ((-1.99*1e-6)*((t)**2)*((h)**2))
    return HIF

def fahrenheit(c):

    """
    Convert a temperature from Celsius to Fahrenheit.

    Args:
        c (float): The temperature in degrees Celsius.

    Returns:
        float: The temperature in degrees Fahrenheit.

    """

    return (c*(9/5)) + 32

def celcius(f):

    """
    Convert a temperature from Fahrenheit to Celsius.

    Args:
        f (float): The temperature in degrees Fahrenheit.

    Returns:
        float: The temperature in degrees Celsius.

    """

    return (f - 32)*(5/9)

def heatwave_ky(temp, t_max, t_min):

    """
    Identify heatwaves based on temperature thresholds (according to Kysely (2010)).

    Args:
        temp (pandas.Series): Series containing temperature values.
        t_max (float): Maximum temperature threshold for heatwave.
        t_min (float): Minimum temperature threshold for heatwave.

    Returns:
        pandas.Series: Series indicating the presence of heatwaves (1) or not (0).

    """

    gtmax = temp >= t_max
    gtmin = temp >= t_min
    days_max = 0
    days_min = 0
    n = len(temp)
    h = [0]*n
    for i in range(n):
        lookback = np.all(gtmax.iloc[(i-(days_max + days_min)):(i-days_min)])
        if (not gtmin.iloc[i]) & (days_max < 3):
            days_max = 0
            days_min = 0
        elif (gtmax.iloc[i]) & (lookback):
            days_max += 1
        elif (not gtmax.iloc[i]) & (days_max >= 3) & (lookback):
            avg = temp.iloc[(i-(days_max + days_min)):(i+1)].mean()
            if (avg >= t_max) & (gtmin.iloc[i]):
                days_min += 1
            else:
                h[(i-(days_max + days_min)):(i)] = [1 for i in range((days_max + days_min))]
                days_max = 0
                days_min = 0
    return pd.Series(h)

def indexer(ti):

    """
    Create an index for heatwave identification.

    Args:
        ti (pandas.Series): Time index.

    Returns:
        list: List containing the index values.

    """

    n = len(ti)
    f = []
    start = 0
    for i in range(n):
        if ti.iloc[i] > 1:
            start += 1
        f.append(start)
    return f
# %%
class HeatwaveM(pd.DataFrame):

    """
    Custom DataFrame class for heatwave analysis.

    Inherits from pandas.DataFrame.

    """

    def  __init__(self,input):

        """
        Initialize the HeatwaveM object.

        Args:
            input (pandas.DataFrame): Input DataFrame.

        """

        df = input.reset_index(drop=True)
        df.columns = ['STATION_ID','TIME','TEMP','HUMID']
        df['DATE'] = df['TIME'].dt.date
        df['HEATWAVE'] = np.nan
        df['IND'] = np.nan
        df = df[['STATION_ID','TIME','DATE','TEMP','HUMID','HEATWAVE','IND']]
        super(HeatwaveM, self).__init__(df)
        self.celcius = True
        self.apparent = False
 
    def to_apparent_nws(self):

        """
        Convert temperature values to apparent temperature using the NWS formula.

        Raises:
            ValueError: If temperature values are already given as apparent temperature.

        """

        if not self.apparent:
            self['TEMP']=self.apply(lambda x: apparent_temperature_nws(x['TEMP'],x['HUMID']), axis=1)
            self.apparent = True
        else:
            raise ValueError('Temperature already given as apparent temperature. Thus, no transformation applied.')
            
    def to_apparent_std(self):

        """
        Convert temperature values to apparent temperature using the Steadman formula.

        Raises:
            ValueError: If temperature values are already given as apparent temperature.

        """

        if not self.apparent:
            self['TEMP']=self.apply(lambda x: apparent_temperature_std(x['TEMP'],x['HUMID']), axis=1)
            self.apparent = True
        else:
            raise ValueError('Temperature already given as apparent temperature. Thus, no transformation applied.')

    def to_celcius(self):

        """
        Convert temperature values to degrees Celsius.

        Raises:
            ValueError: If temperature values are already given in Celsius.

        """

        if not self.celcius:
            self['TEMP']=self.apply(lambda x: celcius(x['TEMP']), axis=1)
            self.celcius = True
        else:
            raise ValueError('Temperature already given in Celcius. Thus, no transformation applied.')
         
    def to_fahrenheit(self):

        """
        Convert temperature values to degrees Fahrenheit.

        Raises:
            ValueError: If temperature values are already given in Fahrenheit.

        """

        if self.celcius:
            self['TEMP']=self.apply(lambda x: fahrenheit(x['TEMP']), axis=1)
            self.celcius = False
        else:
            raise ValueError('Temperature already given in Fahrenheit. Thus, no transformation applied.')

    
    def check(self,kind="both"):

        """
        Check the temperature unit and apparent temperature status.

        Args:
            kind (str): Type of check to perform. Options: "both", "celcius", "apparent".

        """

        if kind=="both":
            print("celcius: {}\napparent: {}".format(self.celcius, self.apparent))
        elif kind=="celcius":
            print("celcius: {}".format(self.celcius))
        elif kind=="apparent":
            print("apparent: {}".format(self.apparent))     

    def get_heatwaves_ky(self,station_id=3379,year=[2022],t_max=30,t_min=25):

        """
        Identify heatwaves based on temperature thresholds (Kentucky method).

        Args:
            station_id (int): Station ID for heatwave analysis.
            year (list): List of years for heatwave analysis.
            t_max (float): Maximum temperature threshold for heatwave.
            t_min (float): Minimum temperature threshold for heatwave.

        """

        n = self[(self['STATION_ID'] == station_id) & (self['TIME'].dt.year.isin(year))]
        sub = n.groupby(n['DATE'], as_index=False).max()[['DATE', 'TEMP']]
        sub['HEATWAVE'] = heatwave_ky(temp=sub['TEMP'], t_max=t_max, t_min=t_min)
        sub = sub[sub['HEATWAVE'] == 1]
        sub['DELTA'] = (sub['DATE'] - sub['DATE'].shift()).dt.days.fillna(1)
        sub['IND'] = indexer(sub['DELTA'])
        sub['HEATWAVE'] = sub['HEATWAVE']
        sub = n[['STATION_ID','TIME','DATE','TEMP','HUMID']].merge(
            sub[['DATE','HEATWAVE','IND']],
            how="left", on="DATE"
        )
        n = n.reset_index(drop=True)
        n['HEATWAVE'] = sub['HEATWAVE'].fillna(0).reset_index(drop=True)
        n['IND'] = sub['IND'].reset_index(drop=True)
        df = n[['STATION_ID','TIME','DATE','TEMP','HUMID','HEATWAVE','IND']]
        super(HeatwaveM, self).__init__(df)

    def heatwaves_summary(self):

        """
        Generate a summary of identified heatwaves.

        Returns:
            pandas.DataFrame: Summary of heatwaves, including minimum and maximum temperatures.

        Raises:
            ValueError: If no heatwaves have been identified.

        """

        sub = self[self['HEATWAVE'] == 1]
        if sub.empty:
            raise ValueError('Either heatwaves have not yet been identified via `get_heatwaves_ky` or no heatwaves have been found for the specified period.')
        else:
            return sub.groupby(['IND','DATE']).agg(["min","max"])[['TEMP']]

    def save_heatwaves_to_list(self,name='heatwaves.pkl'):

        """
        Save the dates of identified heatwaves to a pickle file.

        Args:
            name (str): Name of the output file.

        """
        
        sub = self[self['HEATWAVE'] == 1]['DATE'].unique().tolist()
        with open(config['data']['dwd'] + '/' + name, 'wb') as f:
            pickle.dump(sub, f)