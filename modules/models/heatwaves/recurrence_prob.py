# # %% time series kysely data prepration
# tseries = dwd[(dwd['STATIONS_ID'] == 3379) & (dwd['MESS_DATUM'].dt.year.isin([2019,2020,2021,2022]))].groupby([dwd['MESS_DATUM'].dt.date], as_index=False).max()[['MESS_DATUM', 'TT_TU']]
# tseries_train = tseries[tseries['MESS_DATUM'].dt.year <= 2021]
# tseries_train.set_index('MESS_DATUM', inplace=True)
# tseries_test = tseries[tseries['MESS_DATUM'].dt.year == 2022]
# tseries_test.set_index('MESS_DATUM', inplace=True)
# # %%
# tt_test = tseries_test.diff(1)
# tt_test = tt_test.fillna(method="bfill")
# plt.figure(figsize=(20,10))
# tt_test.plot()
# # %%
# tt = tseries_train.diff(1)
# tt = tt.fillna(method="bfill")
# plt.figure(figsize=(20,10))
# tt.plot()
# # %% dickey fuller test
# stattools.adfuller(
#     tt,
#     regression="c",
#     autolag="AIC"
# )
# # %% arma model
# arma_mod = ARIMA(tt, order=(2,0,2)).fit()
# # %%
# print(arma_mod.summary())
# # %%
# ar = arma_mod.arparams
# ma = arma_mod.maparams
# ar = np.r_[1, -ar]
# ma = np.r_[1, ma]
# # %%
# mat = np.empty([1000,365])
# for i in range(1000):
#     mat[i,:] = arma_generate_sample(ar, ma, nsample=365)
# result = np.empty([1000,365])
# for i in range(1000):
#     result[i,:] = np.reshape(np.where(tt_test > np.reshape(mat[i,:],[-1,1]),1,0),[365,])
# means = result.mean(axis=0)
# pd.Series(means).plot(kind="bar")