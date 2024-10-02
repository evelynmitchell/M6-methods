import os.path

import numpy as np
import pandas as pd

import statsmodels.api as sma
import datetime as dt
import xarray as xr
from numpyro import distributions as dist
from tqdm import tqdm

def calc_cov(dt,factor_def,asset_data,gamma,omega):

    n_factors=len (factor_def.columns)
    returns = asset_data[asset_data.index<=dt].tail(-3).dropna(how='all')
    returns = (returns - returns.mean(0))/returns.std(0)
    l0_factors = returns.reindex(factor_def.index, axis=1).fillna(0) @ factor_def[fl0]
    l0_factors = (l0_factors-l0_factors.mean())/l0_factors.std()

    #estimate shinkage target (single index model) cov_mtx
    idxs = {i: returns[i].loc[l0_factors['ALL'].index].dropna().index for i in returns.columns}
    idxs = {i: idxs[i] for i in idxs.keys() if len(idxs[i]) != 0}
    models = {i: sma.OLS(returns.loc[idxs[i], i],
                         sma.add_constant(l0_factors['ALL'].loc[idxs[i]])).fit() for i in idxs.keys()}
    l0_loadings = pd.DataFrame(np.asarray([m.params[1] for m in models.values()]),index=idxs.keys())

    coords = {'t': l0_factors.index, 'x': ['ALL'], 'y': etfs}
    dims = coords.keys()
    fcov = xr.DataArray(np.nan, coords=coords, dims=dims)
    #dynamic covariance of l0 factor with all ETFs
    fcov[0] = l0_loadings.reindex(etfs).T
    for ti in range(1, len(l0_factors.index)):
        fcov[ti] = (((1 - gamma - omega) * fcov[ti - 1]
                    + omega * l0_loadings.reindex(etfs).T
                    + gamma * np.outer(l0_factors.iloc[ti],returns.iloc[ti].reindex(etfs))))
        fcov[ti] = np.where(np.isnan(fcov[ti]),fcov[ti-1],fcov[ti])

    #residuals from l0 process
    l0_residuals = (returns[etfs]-(fcov.values*l0_factors.values[...,np.newaxis]).squeeze())
    l0_residuals = (l0_residuals-l0_residuals.mean())/l0_residuals.std()

    #l1 factors
    l1_factors = l0_residuals.reindex(factor_def.index, axis=1).fillna(0) @ factor_def[fl1]
    l1_factors = (l1_factors-l1_factors.mean())/l1_factors.std()

    #factor covariance matrix. shrinkage target is now identity
    coords = {'t': l0_factors.index, 'x': factor_def.columns, 'y': factor_def.columns}
    dims = coords.keys()
    fcov = xr.DataArray(np.nan, coords=coords, dims=dims)
    fcov[0] = np.eye(n_factors)
    for ti in range(1, len(l0_factors.index)):
        t = l0_factors.index[ti]
        fret = pd.concat([l0_factors.iloc[ti], l1_factors.iloc[ti]], axis=0)
        fcov[ti] = (((1 - gamma - omega) * fcov[ti - 1]
                     + omega * np.eye(n_factors)
                     + gamma * np.outer(fret, fret)))
        scaling_factor = np.sqrt(1 / np.diag(fcov[ti]))
        fcov[ti] = np.diag(scaling_factor) @ fcov[ti].values @ np.diag(scaling_factor)
    #factor / asset cov_mtxet
    #shrinkage target

    #loadings_all = pd.concat([pd.Series(np.eye(n_factors)[:,0],factor_def.columns),l0_loadings],0)
    #idxs = {i: returns[i].loc[l0_factors['ALL'].index].dropna().index for i in etfs}
    #idxs = {i: idxs[i] for i in idxs.keys() if len(idxs[i]) != 0}
    #models = {i: sma.OLS(returns.loc[idxs[i], i],
    #                     l0_factors['ALL'].loc[idxs[i]]).fit() for i in idxs.keys()}
    #l0_loadings = pd.DataFrame(np.asarray([m.params[0] for m in models.values()]),index=idxs.keys())
    # shrunk_cov_mtx = np.outer(l0_loadings,l0_loadings) + np.diag(spec_var_all)
    # shrunk_cov_mtx = np.diag(np.sqrt(1 / np.diag(shrunk_cov_mtx))) @ shrunk_cov_mtx @ np.diag(np.sqrt(1 / np.diag(shrunk_cov_mtx)))
    # loadings of factors and assets
    # loadings_all = pd.concat([pd.Series(np.eye(n_factors)[:, 0], factor_def.columns), l0_loadings], axis=0)
    #cov_est_0 = shrunk_cov_mtx[:n_factors,n_factors:]

    run_dates = returns.index
    factors = pd.concat([l0_factors, l1_factors], axis=1)
    spec_var_all = np.asarray([(m.resid**2).sum()/len(m.resid) for m in models.values()])
    assets = list(idxs.keys())
    coords = {'t': run_dates, 'factor': factor_def.columns, 'asset': assets}
    dims = coords.keys()
    loadings = xr.DataArray(np.nan,coords=coords,dims=dims)
    spec_var = pd.DataFrame(np.nan,index=run_dates,columns=assets)

    loadings_all = pd.concat([pd.Series(np.eye(n_factors)[:, 0], factor_def.columns), l0_loadings], axis=0)
    cov_est_0 = np.outer(loadings_all,loadings_all)[:n_factors,n_factors:]
    loadings_0 = np.linalg.pinv(fcov[0])@cov_est_0
    cov_est = cov_est_0
    loadings[0] = loadings_0
    spec_var_0 = spec_var_all
    spec_var.iloc[0] = spec_var_0

    for ti in range(1,len(returns.index)):
        cov_est = ((1 - gamma - omega) * cov_est
                   + omega * cov_est_0
                   + gamma * np.outer(factors.loc[run_dates[ti]], returns.loc[run_dates[ti],assets]))
        cov_est = np.where(np.isnan(cov_est),cov_est_0,cov_est)
        loadings[ti] = np.linalg.pinv(fcov[ti])@cov_est

        # specific variance
        err = returns.loc[run_dates[ti]][assets] - factors.loc[run_dates[ti]] @ loadings[ti].values
        spec_var.iloc[ti] = ((1 - gamma - omega) * spec_var.iloc[ti - 1]
                             + omega * spec_var_0
                             + gamma * err ** 2)
        spec_var.iloc[ti] = np.where(np.isnan(spec_var.iloc[ti]), spec_var_0, spec_var.iloc[ti])

    #scaling
    for ti in range(1, len(returns.index)):
        scale_factor = np.sqrt(
            np.diag(loadings[ti].values.T @ fcov[ti].values @ loadings[ti].values + np.diag(spec_var.iloc[ti])))
        loadings[ti] = loadings[ti] / scale_factor
        spec_var.iloc[ti] = spec_var.iloc[ti] / scale_factor**2

    return fcov, loadings, spec_var

def log_prob_calc(test_periods,test_period_windows):

    grid = pd.DataFrame(0.0, index=[.03, .02, .01, .005], columns=[.0025, .005, .0075])

    for key in tqdm(test_periods.keys()):

        sd = test_periods[key]
        vols1d = pd.read_hdf(key + '.h5', 'vols1D')
        vol20d = pd.read_hdf(key + '.h5', 'vol20D')

        stn_retn = daily_rtn[daily_rtn.index <= sd] / vols1d[vols1d.index <= sd]

        for gamma in grid.index:
            for omega in grid.columns:
                fcov, loadings, spec_var = calc_cov(sd, factor_def, stn_retn, gamma=gamma, omega=omega)
                scaler = vol20d.loc[sd]
                loadings_final = loadings[-1].to_pandas()
                spec_var_final = spec_var.iloc[-1]
                loadings_final_scaled = loadings_final * scaler[loadings_final.columns].values
                spec_vol_final = np.sqrt(spec_var_final) * scaler[spec_var_final.index].values.squeeze()

                #sys_var = loadings_final_scaled.values.T @ fcov[-1].values @ loadings_final_scaled.values
                #spec_var = spec_vol_final ** 2
                #cov_mtx = pd.DataFrame(sys_var + np.diag(spec_var), daily_rtn.columns, daily_rtn.columns)

                obs = daily_rtn.reindex(test_period_windows[key]).sum().loc[spec_var_final.index]*252/20

                cov_factor = loadings_final_scaled.T @ np.linalg.cholesky(fcov[-1].to_pandas())
                log_Score = dist.LowRankMultivariateNormal(loc=np.zeros(len(spec_var_final.values)),
                                                           cov_factor=cov_factor.values,
                                                           cov_diag=spec_vol_final.values ** 2).log_prob(obs.values)
                grid.loc[gamma, omega] = grid.loc[gamma, omega] + log_Score

    return grid

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


rs_spec=pd.offsets.Week(weekday=4)
universe=pd.read_csv('M6_Universe.csv')
universe['symbol'] = [i[:-2] if i[-2:]=='.L' else i for i in universe['symbol']]
prices = pd.read_hdf('prices.h5', 'prices')

ACt = prices[['Symbol', 'Adj Close']].set_index(['Symbol'], append=True).unstack()
ACt.columns = ACt.columns.droplevel(0)
ACt.columns = [i[:-2] if i[-2:]=='.L' else i for i in ACt.columns]
daily_rtn = np.log(ACt.astype(np.float64)).diff()

etfs = list(universe['symbol'][universe['class']=='ETF'])
factor_def = pd.read_excel('factor_def.xlsx', index_col=0)
factor_def = (factor_def.drop('Name', axis=1)).fillna(0)
factor_def.index = [i[:-2] if i[-2:] == '.L' else i for i in factor_def.index]

fl0 = ['ALL']
fl1 = ['USE', 'EUE', 'AE', 'TERM', 'CREDIT', 'MOM', 'VAL', 'SIZ']
fl1 = fl1 + [i for i in factor_def.columns if i not in fl0 and i not in fl1]


#grid calculation
calc_grid = False
if calc_grid:
    cov_test_pd = pd.date_range(end = dt.datetime(2021,11,30), freq= 4*rs_spec, periods = 36)
    test_periods = {'Test Period'+str(i):cov_test_pd[i] for i in range(len(cov_test_pd))}
    test_period_windows = {tl: pd.bdate_range(start = test_periods[tl],periods=21)[1:] for tl in test_periods.keys()}
    log_probs = log_prob_calc(test_periods,test_period_windows)

#risk model for calculations
calc_models = True
omega = .005
gamma = .01

if calc_models:

    for key in tqdm(submission_dates.keys()):

        sd = submission_dates[key]
        vols1d = pd.read_hdf(key + '.h5', 'vols1D')
        vols1d.columns = [i[:-2] if i[-2:] == '.L' else i for i in vols1d.columns]
        vol20d = pd.read_hdf(key + '.h5', 'vol20D')
        vol20d.columns = [i[:-2] if i[-2:] == '.L' else i for i in vol20d.columns]

        stn_retn = daily_rtn[daily_rtn.index <= sd] / vols1d[vols1d.index <= sd]
        fcov, loadings, spec_var = calc_cov(sd, factor_def, stn_retn, omega=omega, gamma=gamma)


        scaler = vol20d.iloc[-1]
        loadings_final = loadings[-1].to_pandas()
        spec_var_final = spec_var.iloc[-1]
        loadings_final_scaled = loadings_final * scaler[loadings_final.columns].values
        spec_vol_final = np.sqrt(spec_var_final) * scaler[spec_var_final.index].values.squeeze()

        #sys_var = loadings_final_scaled.values.T@fcov[-1].values@loadings_final_scaled.values
        #spec_var = spec_vol_final**2
        #cov_mtx = pd.DataFrame(sys_var + np.diag(spec_var),daily_rtn.columns,daily_rtn.columns)

        #sys_var = loadings_final.values.T @ fcov[-1].values @ loadings_final.values
        #spec_var = spec_var_final
        #cov_mtx = pd.DataFrame(sys_var + np.diag(spec_var), daily_rtn.columns, daily_rtn.columns)

        loadings_final_scaled.to_csv(key+'_factor_loadings.csv')
        spec_vol_final.to_csv(key + '_specific_vol.csv')
        fcov[-1].to_pandas().to_csv(key+'_factor_cov.csv')