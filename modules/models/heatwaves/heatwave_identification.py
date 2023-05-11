#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import statsmodels.tsa.stattools as s
from statsmodels.tsa.arima.model import ARIMA
config_path = '/home/tu/tu_tu/tu_zxmny46/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% time series analysis
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')
dwd['MESS_DATUM'] = pd.to_datetime(dwd['MESS_DATUM'], format='%Y-%m-%d %H')
dwd.head()
#%% functions
def apparent_temperature_nws(r):
    if r['TT_TU_FA'] <= 40:
        HIF = r['TT_TU_FA']
    else:
        A = -10.3 + 1.1*r['TT_TU_FA'] + 0.047*r['RF_TU']
        if A < 79:
            HIF = A
        else:
            B = (-42.379) + ((2.04901523)*r['TT_TU_FA']) + ((10.14333127)*r['RF_TU']) + ((-0.22475541)*r['TT_TU_FA']*r['RF_TU']) + ((-6.83783*1e-3)*((r['TT_TU_FA'])**2)) + ((-5.481717*1e-2)*((r['RF_TU'])**2)) + ((1.22874*1e-3)*((r['TT_TU_FA'])**2)*r['RF_TU']) + ((8.5282*1e-4)*r['TT_TU_FA']*((r['RF_TU'])**2)) + ((-1.99*1e-6)*((r['TT_TU_FA'])**2)*((r['RF_TU'])**2))
            if (r['RF_TU'] <= 13) & (r['TT_TU_FA'] >= 80) & (r['TT_TU_FA'] <= 112):
                HIF = B - ((13- r['RF_TU'])/4)*(((17 - abs(r['TT_TU_FA'] - 95))/17)**0.5)
            elif (r['RF_TU'] > 85) & (r['TT_TU_FA'] >= 80) & (r['TT_TU_FA'] <= 87):
                HIF = B + 0.02*(r['RF_TU'] - 85)*(87 - r['TT_TU_FA'])
            else:
                HIF = B
    return HIF

def apparent_temperature_std(r):
    if r['TT_TU_FA'] <= 80:
        HIF = r['TT_TU_FA']
    elif r['RF_TU'] <= 40:
        HIF = r['TT_TU_FA']
    else:
        HIF = (-42.379) + ((2.04901523)*r['TT_TU_FA']) + ((10.14333127)*r['RF_TU']) + ((-0.22475541)*r['TT_TU_FA']*r['RF_TU']) + ((-6.83783*1e-3)*((r['TT_TU_FA'])**2)) + ((-5.481717*1e-2)*((r['RF_TU'])**2)) + ((1.22874*1e-3)*((r['TT_TU_FA'])**2)*r['RF_TU']) + ((8.5282*1e-4)*r['TT_TU_FA']*((r['RF_TU'])**2)) + ((-1.99*1e-6)*((r['TT_TU_FA'])**2)*((r['RF_TU'])**2))
    return HIF

def fahrenheit(c):
    return (c*(9/5)) + 32

def celcius(f):
    return (f - 32)*(5/9)
#%%
dwd['TT_TU_FA'] = dwd.apply(lambda x: fahrenheit(x['TT_TU']), axis=1)
# %%
dwd['TA_STD_FA'] = dwd.apply(apparent_temperature_std, axis=1)
dwd['TA_NWS_FA'] = dwd.apply(apparent_temperature_nws, axis=1)
# %%
dwd['TA_STD'] = dwd.apply(lambda x: celcius(x['TA_STD_FA']), axis=1)
dwd['TA_NWS'] = dwd.apply(lambda x: celcius(x['TA_NWS_FA']), axis=1)
#%% kysely
t_max = 30
t_min = 25
mcity = dwd[(dwd['STATIONS_ID'] == 3379) & (dwd['MESS_DATUM'].dt.year == 2022)].groupby([dwd['MESS_DATUM'].dt.date], as_index=False).max()[['MESS_DATUM', 'TT_TU']]
mcity['GTMAX'] = mcity['TT_TU'] >= t_max
mcity['GTMIN'] = mcity['TT_TU'] >= t_min
mcity['DAYSMAX'] = mcity['GTMAX'].rolling(window=3).sum()
#%% kysely
h = [0]*mcity.shape[0]
days_max = 0
days_min = 0
#%% kysely
for i in range(mcity.shape[0]):
    if (mcity['GTMIN'].iloc[i] == False) & (days_max < 3):
        days_max = 0
        days_min = 0
    elif (mcity['GTMAX'].iloc[i] == True):
        days_max += 1
    elif (mcity['GTMAX'].iloc[i] == False) & (days_max >= 3):
        avg = mcity['TT_TU'].iloc[(i-(days_max + days_min)+1):(i+1)].mean()
        if (avg >= t_max) & (mcity['GTMIN'].iloc[i] == True):
            days_min += 1
        else:
            if mcity['GTMAX'].iloc[(i-days_max):(i-days_min+1)].to_list() == [True, True, True]:
                h[(i-(days_max + days_min)):(i)] = [1 for i in range((days_max + days_min))]
            days_max = 0
            days_min = 0
mcity["HEATWAVE"] = h
# %%
tseries = dwd[(dwd['STATIONS_ID'] == 3379) & (dwd['MESS_DATUM'].dt.year.isin([2019,2020,2021,2022]))].groupby([dwd['MESS_DATUM'].dt.date], as_index=False).max()[['MESS_DATUM', 'TT_TU']]
tseries_train = tseries[tseries['MESS_DATUM'].dt.year <= 2021]
tseries_train.set_index('MESS_DATUM', inplace=True)
tseries_test = tseries[tseries['MESS_DATUM'].dt.year == 2022]
tseries_test.set_index('MESS_DATUM', inplace=True)


# %%
plt.figure(figsize=(20,10))
tseries_test.diff(1).plot()
# %%
plt.figure(figsize=(20,10))
tseries_train.diff(1).plot()
# %%
tt = tseries_train.diff(1)
tt = tt.fillna(method="bfill")
# %%
s.adfuller(
    tt,
    regression="c",
    autolag="AIC"
)
# %%
arma_mod = ARIMA(tt, order=(2,0,2)).fit()
# %%
print(arma_mod.summary())
# %%
ar = arma_mod.arparams
ma = arma_mod.maparams
ar = np.r_[1, -ar]
ma = np.r_[1, ma]
# %%
t_sample = arp.arma_generate_sample(ar, ma, nsample=365)
pd.Series(t_sample).plot()
# %%
mat = np.empty([1000,365])
for i in range(1000):
    mat[i,:] = arp.arma_generate_sample(ar, ma, nsample=365)
# %%
tt_test = tseries_test.diff(1)
tt_test = tt_test.fillna(method="bfill")
# %%
result = np.empty([1000,365])
for i in range(1000):
    result[i,:] = np.reshape(np.where(tt_test > np.reshape(mat[i,:],[-1,1]),1,0),[365,])
# %%
i = 1
np.where(tt_test > np.reshape(mat[i,:],[-1,1]),1,0)
# %%
means = result.mean(axis=0)
# %%
