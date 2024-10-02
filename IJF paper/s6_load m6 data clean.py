import os.path

import numpy as np
import pandas as pd
import statsmodels.api as sma
import datetime as dt

from matplotlib.pyplot import xlabel
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.ticker as mtick
sns.set_theme()

#import pandas_datareader as pdr
#import yfinance as yf
#yf.pdr_override()

def get_historic_data(ticker,sd,ed):
    return pdr.get_data_yahoo(ticker, start=sd, end=ed)

def load_history(tiks,sd,ed):
    for tk in tiks:
        print(tk)
        if not os.path.exists(tk + '.h5'):
            print('Downloading')
            data = get_historic_data(tk, sd, ed)
            data.to_hdf(tk + '.h5', 'prices')
        else:
            print('Found')

def assemble_price_table(tiks,sd,ed):
    #data = pd.DataFrame(index=pd.bdate_range(sd,ed),columns=tiks)
    prices = pd.DataFrame()
    for tk in tiks:
        print (tk)
        tkdata = pd.read_hdf(tk + '.h5')
        #data[tk] = tkdata['Adj Close']
        tkdata['Symbol'] = tk
        prices=pd.concat([prices,tkdata],axis=0)
    prices.to_hdf('prices.h5','prices')

def clean_prices(prices, period_def, field):
    #fills forward any missing values
    #i.e. does not fill in historic 'missing' values where data does not exist for a particular series
    locator = [i for i in period_def if i in prices.index]
    tmp =  prices.loc[locator,['Symbol',field]].set_index(['Symbol'],append=True).unstack().reindex(period_def).fillna(method='ffill')
    tmp.columns = tmp.columns.droplevel(0)
    return tmp

def calc_variance(base_period,prices,rs_spec):
    Ot = clean_prices(prices,base_period,'Open')
    Ct = clean_prices(prices, base_period, 'Close')
    Ctm1 = Ct.shift(1)
    Ht = clean_prices(prices, base_period, 'High')
    Lt = clean_prices(prices, base_period, 'Low')
    ot = (np.log(Ot)-np.log(Ctm1))
    ut = (np.log(Ht)-np.log(Ot))
    dt = (np.log(Lt)-np.log(Ot))
    ct = (np.log(Ct)-np.log(Ot))
    #close to close variance
    Vcc = np.log(Ct).diff()**2
    #single period GK calcs
    Vrs = ut*(ut-ct)+dt*(dt-ct)
    Vp = (ut-dt)**2 / (4 * np.log(2))
    Vto = ot**2
    Vtc = ct**2
    Vgk = Vto -0.383 * Vtc + 1.364 * Vp + 0.019 * Vrs
    #multi period YZ Variance values
    Vrs_m = Vrs.resample(rs_spec).mean()
    npds = ot.resample(rs_spec).count()
    df_adjust = npds / (npds - 1)
    Vo = ((ot - ot.resample(rs_spec).mean().reindex(ot.index).fillna(method='bfill'))**2).resample(rs_spec).mean() * df_adjust
    Vc = ((ct - ct.resample(rs_spec).mean().reindex(ct.index).fillna(method='bfill'))**2).resample(rs_spec).mean() * df_adjust
    k = .34/(1.34 + (npds+1)/(npds-1))
    Vyz  = Vo + k * Vc + (1-k) * Vrs_m
    Vcc_pp = Vcc.resample(rs_spec).mean()
    #entire period expanding YZ Variance
    Vrs_m = Vrs.expanding().mean()
    npd = Vrs.expanding().count()
    Vo = ((ot - ot.expanding().mean()) ** 2).mean()*npd/(npd-1)
    Vc = ((ct - ct.expanding().mean()) ** 2).mean()*npd/(npd-1)
    k = .34 / (1.34 + (npd + 1) / (npd - 1))
    Vyz_mean = Vo + k * Vc + (1 - k) * Vrs_m
    Vcc_mean = Vcc.expanding().mean()
    return Vgk, Vcc, Vyz, Vcc_pp, Vyz_mean, Vcc_mean

def vol_model(my_items):

    #MY ITEMS ARE THE CALCULATION PERIODS

    Vgk, Vcc, Vyz, Vcc_pp, Vyz_mean, Vcc_mean = calc_variance(full_period, prices, rs_spec)
    # Vgk is the daily Garman Klass realised variance
    # Vyz is the Yang / Zhang realised variance for the aggregate periods
    # Vyz_mean is the long run asset yz variance
    # Use this as dependent variable in the regression
    # these are all on a 'daily' scale

    RV_daily = np.sqrt(Vgk)
    RV_actual = np.sqrt(Vyz)
    RVlr = np.sqrt(Vyz_mean)

    # GlRV
    GlRV = RVlr.multiply((RV_daily / RVlr).mean(1), 0)

    # 20 day regression
    # dependent varibles, shift forward 1 day
    Y_20 = (RV_actual - RVlr.loc[RV_actual.index]).tail(-1)
    Y_1 = (RV_daily - RVlr).tail(-1)
    RV1 = (RV_daily.ewm(com=1).mean() - RVlr).shift(1)
    RV5 = (RV_daily.ewm(com=5).mean() - RVlr).shift(1)
    RV25 = (RV_daily.ewm(com=25).mean() - RVlr).shift(1)
    RV125 = (RV_daily.ewm(com=125).mean() - RVlr).shift(1)
    RVG5 = (GlRV.ewm(com=5).mean() - RVlr).shift(1)
    exp_vars = [RV1, RV5, RV25, RV125, RVG5]
    exp_var_names = ['RV1', 'RV5', 'RV25', 'RV125', 'RVG5']

    vols20D = pd.DataFrame(dtype=np.float64,columns=Y_20.columns)

    # fit the models
    for c in my_items:

        print('Calculating',c)

        calc_date = my_items[c]
        ofile = pd.HDFStore(c +'.h5')

        #20 day model
        Y_20_fit = Y_20[Y_20.index <= calc_date].stack()
        ivars = pd.concat([i.loc[RV_actual.index].stack().loc[Y_20_fit.index] for i in exp_vars], axis=1). \
            reindex(Y_20_fit.index).set_axis(exp_var_names, axis=1).dropna(how='any')
        Y_20_fit = Y_20_fit.reindex(ivars.index)
        model = sma.OLS(Y_20_fit, ivars).fit()
        params = model.params
        ivars = pd.concat([i.loc[RV_actual.index][i.loc[RV_actual.index].index <= calc_date].tail(1) for i in exp_vars],
                          axis=0).set_axis(exp_var_names, axis=0).T
        vols20D = pd.concat([vols20D,(ivars @ params + RVlr[RVlr.index <= calc_date].shift(1).tail(1))*np.sqrt(252)],axis=0)

        #20 day values are annualised
        ofile.put('vol20D',vols20D)

        #ofile.put('last_obs_vol',Y_20[Y_20.index <= calc_date].tail(1))

        #1 day model
        Y_1_fit = Y_1[Y_1.index <= calc_date].stack()
        ivars = pd.concat([i.stack().reindex(Y_1_fit.index) for i in exp_vars],
                          axis=1).set_axis(exp_var_names, axis=1).dropna(how='any')
        Y_1_fit = Y_1_fit.reindex(ivars.index)
        model_1D = sma.OLS(Y_1_fit, ivars).fit()
        params = model.params
        ivars = pd.concat([i.stack() for i in exp_vars],
                          axis=1).set_axis(exp_var_names, axis=1).dropna(how='any').reindex(Y_1_fit.index)
        vols1D = (ivars@params).unstack()+RVlr[RVlr.index <= calc_date].shift(1)

        ofile.put('vols1D',vols1D)

        ofile.close()

    # capture model weights and asset scores at each time t

def get_prices():
    prices = pd.read_hdf('prices.h5', 'prices')





#INITIAL SETUP
universe=pd.read_csv('M6_Universe.csv')
sd = dt.datetime(2012,2,17)
ed = dt.datetime(2023,2,17)
#load_history(universe['symbol'].to_list(),sd,ed)
assemble_price_table(universe['symbol'].to_list(),sd,ed)


#ESTIMATE VOL MODEL
prices=pd.read_hdf('prices.h5','prices')
ACt = prices[['Symbol','Adj Close']].set_index(['Symbol'],append=True).unstack()
#spec for calculating target RV
rs_spec=4*pd.offsets.Week(weekday=4)
full_period_est = pd.bdate_range(end=dt.datetime(2023,2,3),freq=rs_spec,periods=144)
full_period = pd.bdate_range(freq='B',start=full_period_est[0], end=dt.datetime(2023,2,3))
submission_dates = {'Trial Period':dt.datetime(2022,2,6),
                    '1st Submission': dt.datetime(2022,3,6),
                    '2nd Submission': dt.datetime(2022,4,3),
                    '3rd Submission': dt.datetime(2022,5,1),
                    '4th Submission': dt.datetime(2022,5,29),
                    '5th Submission': dt.datetime(2022,6,26),
                    '6th Submission': dt.datetime(2022,7,24),
                    '7th Submission': dt.datetime(2022,8,21),
                    '8th Submission': dt.datetime(2022,9,18),
                    '9th Submission': dt.datetime(2022,10,16),
                    '10th Submission' :dt.datetime(2022,11,13),
                    '11th Submission': dt.datetime(2022,12,11),
                    '12th Submission': dt.datetime(2023,1,8)}
cov_test_pd = pd.date_range(end = dt.datetime(2021,11,30), freq= rs_spec, periods = 36)
ACt.columns = ACt.columns.droplevel(0)
ACt = ACt.reindex(full_period).fillna(method='ffill')
daily_rtn = np.log(ACt.astype(np.float64)).diff()
vol_model(submission_dates)
vol_model({'Test Period'+str(i):cov_test_pd[i] for i in range(len(cov_test_pd))})


#chart of normalised volatility
Vgk, Vcc, Vyz, Vcc_pp, Vyz_mean, Vcc_mean = calc_variance(full_period, prices, rs_spec)
RV_daily = np.sqrt(Vgk)
RV_actual = np.sqrt(Vyz)
RVlr = np.sqrt(Vyz_mean)
Y_20 = (RV_actual / RVlr.loc[RV_actual.index])
Y_1 = (RV_daily / RVlr)
assets = ['IVV', 'GSG', 'IEUS', 'LQD']
knl=[gaussian_kde(Y_20[i].fillna(0)) for i in assets]
kdes = pd.DataFrame([i(np.linspace(0,4,50)) for i in knl]).T
kdes.index = np.linspace(0,4,50)
kdes.columns = assets
kdes.plot(xlabel='Daily RV / Long run mean')

