import os.path

import numpy as np
import pandas as pd
#from pandas_datareader import data as pdr
import statsmodels.api as sma
import datetime as dt
import xarray as xr
from numpyro import distributions as dist
from tqdm import tqdm, trange
import jax.numpy as jnp
from jax import jit, grad
import scipy.optimize as scopt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
sns.set_theme()

def optimise(pd_key, rs, ic, risk=None):
    loadings_d = loadings[pd_key]
    spec_var_d = spec_risk[pd_key] ** 2
    fcov_d = fcov[pd_key]
    cov_mtx = loadings_d.values.T @ fcov_d.values @ loadings_d.values
    vol = np.sqrt(np.diag(cov_mtx) + spec_var_d.T).values.squeeze()

    rs_sd = rs.std()
    rs_sd =jnp.where(rs_sd==0,1.41,rs_sd)

    scores_stn = (rs - rs.mean())/rs_sd

    alphas = vol * ic * scores_stn.fillna(0).values

    args = (alphas, cov_mtx, spec_var_d.values)

    def objective_fn(wts, alphas, cov, spec):
        ret_fc = -wts @ alphas
        rsk_fc = wts @ cov @ wts + (wts ** 2 * spec).sum()
        return ret_fc / rsk_fc

    jit_ob_fn = jit(objective_fn)
    jit_grad_ob_fn = jit(grad(objective_fn))
    spec = spec_var_d.values
    cov = cov_mtx

    if risk is None:

        cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.abs(x).sum()},
            {'type': 'ineq', 'fun': lambda x: np.abs(x).sum() - .25})

    else:

        cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.abs(x).sum()},
                {'type': 'ineq', 'fun': lambda x: np.abs(x).sum() - .25},
                {'type': 'ineq', 'fun': lambda x: x @ cov @ x + (x ** 2 * spec).sum() - risk})


    opt_res = scopt.minimize(fun=jit_ob_fn,
                             x0=np.ones_like(alphas) / 100,
                             args=args,
                             method='SLSQP',
                             jac=jit_grad_ob_fn,
                             constraints=cons)

    #print('Risk Forecast:',risk)
    #x = opt_res.x
    #print('Optimised Portfolio Risk Forecast:', np.abs(risk - x @ cov @ x + (x ** 2 * spec).sum()))

    return opt_res.x, opt_res.success

def reverse_optimise(pd_key, wts):
    loadings_d = loadings[pd_key]
    spec_var_d = spec_risk[pd_key] ** 2
    fcov_d = fcov[pd_key]
    cov_mtx = loadings_d.values.T @ fcov_d.values @ loadings_d.values
    vol = np.sqrt(np.diag(cov_mtx) + spec_var_d.T).values.squeeze()

    max_alpha = .3 * vol * 3
    min_alpha = .3 * vol * -3
    bounds = [(min_alpha[i], max_alpha[i]) for i in range(len(max_alpha))]

    args = (wts, cov_mtx, spec_var_d.values)

    def objective_fn(alphas, wts, cov, spec):
        ret_fc = -wts @ alphas
        rsk_fc = wts @ cov @ wts + (wts ** 2 * spec).sum()
        return ret_fc / rsk_fc

    jit_ob_fn = jit(objective_fn)
    jit_grad_ob_fn = jit(grad(objective_fn))

    cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.abs(x).sum()},
            {'type': 'ineq', 'fun': lambda x: np.abs(x).sum() - .25})

    opt_res = scopt.minimize(fun=jit_ob_fn,
                             x0=np.zeros(100,dtype=np.float64),
                             args=args,
                             method='SLSQP',
                             jac=jit_grad_ob_fn,
                             bounds=bounds)

    #print('Risk Forecast:',risk)
    #x = opt_res.x
    #print('Optimised Portfolio Risk Forecast:', np.abs(risk - x @ cov @ x + (x ** 2 * spec).sum()))

    return opt_res.x, opt_res.success

def map_sub(key):
    subm = key[1][0]
    return 'M'+str(subm)

def quant_rank(data,q):
    return np.floor((data.rank(pct=True) - np.finfo(np.float64).eps) * q) + 1

def fill_portfolios(decisions,scores):
    #forward fill 'missing' submissions with previous submissions adjusted for market moves
    #in the intervening periods.
    all_portfolios = decisions.copy()
    all_scores = scores.copy()
    all_portfolios.index = all_portfolios.index.map(lambda x: (x[0],sort_order[x[1]]))
    all_scores.index = all_scores.index.map(lambda x: (x[0], sort_order[x[1]]))
    teams = list(set(decisions.index.levels[0]))
    asset_period_rtns = pd.concat([np.exp(daily_rtn.reindex(submission_os_periods[i]).sum()) for i in submission_dates.keys()],axis=1)
    for team in teams:
        all_decisions = all_portfolios.loc(axis=0)[team]
        #all_team_scores = all_scores.loc(axis=0)[team]
        if len(all_decisions)<12:
            missing = [x for x in range(1,13) if x not in all_decisions.index]
            for m in missing:
                try:
                    start_wts = all_portfolios.loc(axis=0)[team,m-1]
                    pd_rtns = asset_period_rtns[m-1]
                    end_vals = start_wts*pd_rtns
                    end_wts = end_vals/end_vals.abs().sum()
                    all_portfolios.loc(axis=0)[team,m]=end_wts
                except:
                    #previous portfolio not available
                    pass
                try:
                    all_scores.loc(axis=0)[team, m] = all_scores.loc(axis=0)[team, m - 1].values
                except:
                    pass
    rsort_order = {sort_order[i]: i for i in sort_order.keys()}
    all_portfolios.index = all_portfolios.index.map(lambda x: (x[0], rsort_order[x[1]]))
    all_scores.index = all_scores.index.map(lambda x: (x[0], rsort_order[x[1]]))
    return all_portfolios, all_scores


submissions=pd.read_excel('ranks.xlsx','Submissions')
IR=pd.read_excel('ranks.xlsx','IR',index_col=0)
IR.index = IR.index.map(lambda x: x.split()[0])
RPS=pd.read_excel('ranks.xlsx','RPS',index_col=0)
RPS.index = RPS.index.map(lambda x: x.split()[0])

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

submission_os_periods = {i:pd.bdate_range(submission_dates[i],freq='B',periods=20) for i in submission_dates.keys()}

sort_order = {'Trial Period':0,
                    '1st Submission': 1,
                    '2nd Submission': 2,
                    '3rd Submission': 3,
                    '4th Submission': 4,
                    '5th Submission': 5,
                    '6th Submission': 6,
                    '7th Submission': 7,
                    '8th Submission': 8,
                    '9th Submission': 9,
                    '10th Submission' :10,
                    '11th Submission': 11,
                    '12th Submission':12}

submissions['Submission'][submissions['Submission']=='Trial run']='Trial Period'

loadings = {k:pd.read_csv(k+'_factor_loadings.csv',index_col=0) for k in submission_dates.keys()}
spec_risk = {k:pd.read_csv(k+'_specific_vol.csv',index_col=0) for k in submission_dates.keys()}
fcov = {k:pd.read_csv(k+'_factor_cov.csv',index_col=0) for k in submission_dates.keys()}


universe=pd.read_csv('M6_Universe.csv')
universe['symbol'] = [i[:-2] if i[-2:]=='.L' else i for i in universe['symbol']]
etfs = list(universe['symbol'][universe['class']=='ETF'])
stocks = list(universe['symbol'][universe['class']!='ETF'])
prices = pd.read_hdf('prices.h5', 'prices')

ACt = prices[['Symbol', 'Adj Close']].set_index(['Symbol'], append=True).unstack()
ACt.columns = ACt.columns.droplevel(0)
ACt.columns = [i[:-2] if i[-2:]=='.L' else i for i in ACt.columns]
daily_log_rtn = np.log(ACt).diff()
daily_rtn = np.exp(daily_log_rtn)-1

sys_var = {i:loadings[i].values.T@fcov[i].values@loadings[i].values for i in submission_dates.keys()}
spec_var = {i:spec_risk[i]**2 for i in submission_dates.keys()}
cov_mtx = {i:pd.DataFrame(sys_var[i] + np.diag(spec_var[i]),daily_rtn.columns,daily_rtn.columns) for i in submission_dates.keys()}

active_submissions = submissions[submissions['IsActive']==1]
decisions = active_submissions[['Team','Submission','Symbol','Decision']].pivot_table(values='Decision',columns='Symbol',index=['Team','Submission'])
decisions.columns = [i[:-2] if i[-2:] == '.L' else i for i in decisions.columns]

ranks = active_submissions[['Team','Submission','Symbol','Rank1',
       'Rank2', 'Rank3', 'Rank4', 'Rank5']].pivot_table(values=['Rank1',
       'Rank2', 'Rank3', 'Rank4', 'Rank5'],index=['Team','Submission','Symbol'])

scores = (ranks@np.arange(0,5)).unstack(level=2)
scores.columns = [i[:-2] if i[-2:] == '.L' else i for i in scores.columns]

#fill out missing decisions
all_decisions,all_scores = fill_portfolios(decisions,scores)


def run_risk_anal_fn(all_decisions,active_decsions_index):
    risk = pd.DataFrame(0.0,index=all_decisions.index,columns=['ActiveDecision','ANN_VOL_FCAST','WT_ETF','ABS_ETF','WT_STOCK',
                                                           'ABS_STOCK','CASH_NET_EXP','CASH_GROSS_EXP',
                                                           'BETA SP500','VAR_%ALL','VAR_%FACTOR','VAR_%SPEC','VAR_%COV',
                                                           'ann_return','ann_vol','ann_ir','M6_ir',
                                                           'ic','ic_etf','ic_stock'])
    risk.loc[active_decsions_index,'ActiveDecision'] = 1

    for i in trange(len(all_decisions)):
        d = all_decisions.iloc[i]
        k = d.name
        loadings_d = loadings[d.name[1]]
        spec_vol_d = spec_risk[d.name[1]]
        fcov_d = fcov[d.name[1]]

        #risk analysis
        holdings = d.reindex(loadings_d.columns).fillna(0)
        ivv = pd.Series(0.0, holdings.index)
        ivv['IVV'] = 1
        ivv_loadings = ivv@loadings_d.T
        portfolio_loadings = holdings@loadings_d.T
        sys_var_total = portfolio_loadings@fcov_d@portfolio_loadings
        spec_var_total = holdings@np.diag((spec_vol_d**2).squeeze())@holdings
        var_total = sys_var_total + spec_var_total
        vol_fc = np.sqrt(var_total)
        risk.loc[k,'ANN_VOL_FCAST'] = vol_fc
        var_ivv = ivv_loadings@fcov_d@ivv_loadings + ivv@np.diag((spec_vol_d**2).squeeze())@ivv
        risk.loc[k, 'BETA SP500'] = (portfolio_loadings @ fcov_d @ portfolio_loadings) / var_ivv

        #cash expsoures
        risk.loc[k, 'WT_ETF'] = all_decisions.loc[k, etfs].sum()
        risk.loc[k, 'ABS_ETF'] = all_decisions.loc[k, etfs].abs().sum()
        risk.loc[k, 'WT_STOCK'] = all_decisions.loc[k, stocks].sum()
        risk.loc[k, 'ABS_STOCK'] = all_decisions.loc[k, stocks].abs().sum()
        risk.loc[k, 'CASH_NET_EXP'] = all_decisions.loc[k].sum()
        risk.loc[k, 'CASH_GROSS_EXP'] = all_decisions.loc[k].abs().sum()

        #risk decomp
        var_ALL = portfolio_loadings['ALL']**2
        var_factors = portfolio_loadings.drop('ALL')@fcov_d.drop('ALL',axis=1).drop('ALL',axis=0)@portfolio_loadings.drop('ALL')
        var_cov = sys_var_total - var_ALL - var_factors

        risk.loc[k,'VAR_%ALL'] =  var_ALL / var_total
        risk.loc[k, 'VAR_%FACTOR'] =  var_factors / var_total
        risk.loc[k, 'VAR_%SPEC'] =  spec_var_total / var_total
        risk.loc[k, 'VAR_%COV'] = var_cov / var_total

        #return analysis ex post
        sub = all_decisions.index[i]
        #daily returns
        asset_returns = daily_rtn.reindex(submission_os_periods[sub[1]])
        portfolio_return = asset_returns@holdings.fillna(0)
        #continuously compounded
        cc_return = (np.exp(portfolio_return)-1)
        ann_return = np.exp(cc_return.sum() * 252/20) - 1
        ann_vol = np.sqrt(cc_return.var(0)*252)
        ann_ir = ann_return/ann_vol
        risk.loc[k,'ann_vol'] = ann_vol
        risk.loc[k, 'ann_return'] = ann_return
        risk.loc[k, 'ann_ir'] = ann_ir

        #IC analysis
        try:
            my_scores = all_scores.loc[k]
        except:
            print ('oops')
        returns = daily_rtn.reindex(submission_os_periods[k[1]]).sum()
        risk.loc[k, 'ic'] = \
        pd.concat([my_scores.rank().fillna(50), returns.rank().fillna(50)], axis=1).corr().iloc[0, 1]
        risk.loc[k, 'ic_etf'] = \
        pd.concat([my_scores.rank().fillna(50), returns.rank().fillna(50)], axis=1).loc[etfs].corr().iloc[0, 1]
        risk.loc[k, 'ic_stock'] = \
        pd.concat([my_scores.rank().fillna(50), returns.rank().fillna(50)], axis=1).loc[etfs].corr().iloc[0, 1]

    return risk

run_risk_anal = 0
if run_risk_anal:
    risk = run_risk_anal_fn(all_decisions,decisions.index)
    risk.to_hdf('risk_analysis.h5','risk')
else:
    risk=pd.read_hdf('risk_analysis.h5','risk')

def run_opt_fn(all_decisions,risk,all_scores,tgt=False):
    opt_wts_low_ic = pd.DataFrame(0.0, index=all_decisions.index, columns=all_decisions.columns)
    opt_wts_med_ic = pd.DataFrame(0.0, index=all_decisions.index, columns=all_decisions.columns)
    opt_wts_high_ic = pd.DataFrame(0.0, index=all_decisions.index, columns=all_decisions.columns)
    reverse_alpha = pd.DataFrame(0.0, index=all_decisions.index, columns=all_scores.columns)
    for i in trange(len(all_decisions)):
        d = all_decisions.iloc[i]
        k = d.name
        loadings_d = loadings[d.name[1]]
        spec_vol_d = spec_risk[d.name[1]]
        fcov_d = fcov[d.name[1]]
        my_scores = all_scores.loc[k]
        sub = all_decisions.index[i]
        # daily returns
        asset_returns = daily_rtn.reindex(submission_os_periods[sub[1]])

        rev_alphas, opt_code = reverse_optimise(k[1], d.values)
        risk.loc[k, 'Reverse_Opt'] = opt_code
        reverse_alpha.loc[k] = rev_alphas

        # optimisations
        ics = {'L': .05, 'M': .10, 'H': .15}
        for code, ic in ics.items():

            if tgt:
                holdings, opt_code = optimise(k[1], my_scores, ic, risk=risk.loc[k, 'ANN_VOL_FCAST'])
            else:
                holdings, opt_code = optimise(k[1], my_scores, ic)

            portfolio_loadings = holdings @ loadings_d.T
            sys_var_total = portfolio_loadings @ fcov_d @ portfolio_loadings
            spec_var_total = holdings @ np.diag((spec_vol_d ** 2).squeeze()) @ holdings
            var_total = sys_var_total + spec_var_total
            vol_fc = np.sqrt(var_total)
            risk.loc[k, 'OptVol_' + code] = vol_fc
            risk.loc[k, 'OptOK' + code] = opt_code
            portfolio_return = asset_returns @ holdings
            # continuously compounded
            cc_return = (np.exp(portfolio_return) - 1)
            ann_return = np.exp(cc_return.sum() * 260 / 20) - 1
            ann_vol = np.sqrt(cc_return.var(0) * 260)
            ann_ir = ann_return / ann_vol
            risk.loc[k, 'ann_vol_opt_' + code] = ann_vol
            risk.loc[k, 'ann_return_opt_' + code] = ann_return
            risk.loc[k, 'ann_ir_opt_' + code] = ann_ir
            #risk.loc[k, 'M6_ir_opt_' + code] = cc_return.sum() / np.sqrt(cc_return.var())
            if code == 'L':
                opt_wts_low_ic.loc[k] = holdings
            elif code == 'M':
                opt_wts_med_ic.loc[k] = holdings
            else:
                opt_wts_high_ic.loc[k] = holdings

    return risk, opt_wts_low_ic,opt_wts_med_ic,opt_wts_high_ic,reverse_alpha

run_opt = 0
#run optimisations with risk target
tgt=True
if run_opt:

    risk, opt_wts_low_ic, opt_wts_med_ic, opt_wts_high_ic, reverse_alpha = run_opt_fn(all_decisions, risk, all_scores,tgt)

    risk_by_sub = risk.groupby('Submission').mean()
    risk_by_sub.index = [sort_order[k] for k in risk_by_sub.index]

    #info ratio by decile
    risk['ic_quintile'] = quant_rank(risk['ic'],5)
    risk[['ic','ic_quintile','ann_ir','ann_ir_opt_L','ann_ir_opt_M','ann_ir_opt_H']].groupby('ic_quintile').mean().round(2)
    risk[['ic','ic_quintile','ANN_VOL_FCAST','OptVol_L','OptVol_M','OptVol_H']].groupby('ic_quintile').mean().round(2)
    risk[['ic','ic_quintile','ann_vol','ann_vol_opt_L','ann_vol_opt_M','ann_vol_opt_H']].groupby('ic_quintile').mean().round(2)

    if tgt:
        risk.to_hdf('risk_analysis_tgt_optimisations.h5', 'risk')
    else:
        risk.to_hdf('risk_analysis_base_optimisations.h5', 'risk')

    reverse_alpha.to_hdf('reverse_alpha.h5', 'reverse_alpha')

rps_top = RPS.sort_values(['RankGlobal']).head(10)
ir_top = IR.sort_values(['RankGlobal']).head(10)
comb_scores = pd.concat([RPS['RankGlobal'],IR['RankGlobal']],axis=1)
comb_scores.fillna(len(comb_scores)/2,inplace=True)
comb_top = comb_scores.mean(1).sort_values().head(10)
target_team_ids = pd.concat([rps_top,ir_top,comb_top]).index.drop_duplicates()
all_decisions_subset = all_decisions.loc(axis=0)[target_team_ids,:]
top_risk = risk.loc(axis=0)[target_team_ids,:]

def risk_fc_plot():
    #risk forecast plot
    riskvfc = (risk[['ANN_VOL_FCAST','ann_vol']]*100).reset_index(level=1)
    plt = sns.relplot(x="ANN_VOL_FCAST", y="ann_vol", hue="Submission",
                sizes=(40, 400), alpha=.5, palette="muted",
                height=6, data=riskvfc)
    plt.set_axis_labels('Forecast Volatility','Ex Post Volatility')
    plt.fig.suptitle('')
    #fmt = '{:.1%}' # Format you want the ticks, e.g. '40%'
    ticks = mtick.PercentFormatter(decimals=1)
    plt.fig.axes[0].xaxis.set_major_formatter(ticks)
    plt.fig.axes[0].yaxis.set_major_formatter(ticks)
    plt.tight_layout()

def risk_bias_reg():
    riskvfc = (risk[['ANN_VOL_FCAST', 'ann_vol']] * 100).reset_index(level=1)
    sma.OLS(riskvfc['ann_vol'],sma.add_constant(riskvfc['ANN_VOL_FCAST'])).fit().summary()

def bias_reg_by_sub_date():
    riskvfc = (risk[['ANN_VOL_FCAST', 'ann_vol']] * 100).reset_index(level=1)
    res=pd.DataFrame(index=sort_order.keys(),columns=['Bias','S&P500Vol'])
    for x in sort_order.keys():
        riskvfc_tmp = riskvfc[riskvfc['Submission'] == x]
        res.loc[x,'Bias'] = sma.OLS(riskvfc_tmp['ann_vol'], sma.add_constant(riskvfc_tmp['ANN_VOL_FCAST'])).fit().params[0]
        res.loc[x,'S&P500Vol'] = daily_rtn['IVV'].reindex(pd.bdate_range(start=submission_dates[x], freq='B', periods=20)).std()
    return res

def opt(key):
    d = all_decisions.loc[key].squeeze()
    k = d.name
    loadings_d = loadings[d.name[1]]
    spec_vol_d = spec_risk[d.name[1]]
    fcov_d = fcov[d.name[1]]
    my_scores = all_scores.loc[k]
    #sub = all_decisions.index[i]
    # daily returns
    #asset_returns = daily_rtn.reindex(submission_os_periods[sub[1]])
    holdings, opt_code = optimise(k[1], my_scores, .1, risk=risk.loc[k, 'ANN_VOL_FCAST'])

def riskvfc_fn():
    riskvfc = (risk[['ANN_VOL_FCAST', 'ann_vol']] * 100).reset_index(level=1)
    riskvfc_tmp = riskvfc[riskvfc['ann_vol'] <= 10]
    (riskvfc_tmp['ann_vol']-riskvfc_tmp['ANN_VOL_FCAST']).describe()
    (riskvfc['ann_vol']-riskvfc['ANN_VOL_FCAST']).describe()

def ivv_risk_v_fc():
    #S&P500 Risk Forecast v. VXX v. actual
    ivv = pd.Series([cov_mtx[i].loc['IVV','IVV']*np.sqrt(260/20) for i in submission_dates.keys()],submission_dates.keys())
    ivv.index = ivv.index.map(lambda x:sort_order[x])
    vxx = pd.Series([prices[prices['Symbol']=='VXX'].ffill().reindex(submission_os_periods[i])['Open'].tail(1).values.squeeze()
                            for i in submission_dates.keys()],index=submission_dates.keys())
    vxx.index = vxx.index.map(lambda x:sort_order[x])
    ivv_rv = pd.Series([daily_rtn['IVV'].reindex(submission_os_periods[i]).std()*np.sqrt(260) for i in submission_dates.keys()])
    ivv_realised_v_fcast = pd.concat([ivv,vxx/100,ivv_rv],axis=1).astype(np.float64)
    ivv_realised_v_fcast.columns=['Model Forecast','VXX','Realised']
    ax=ivv_realised_v_fcast.plot(ylabel='Annualised Volatility')
    ax.set_xlabel('Submission')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

def risk_by_rps():
    #RPS rankings, NA replaced by 5
    #dunning kreuger?
    RPS_by_month = RPS[['M'+str(i) for i in range(1,13)]]
    RPS_deciles = pd.concat([(RPS_by_month[i].rank()/RPS_by_month[i].count()).fillna(.5).round(1)*10 for i in RPS_by_month.columns],axis=1)
    RPS_rank_by_month = RPS_deciles.stack()

def opt_ports():
    risk_filter = risk[risk['OptOKM']==True]
    risk_filter_2 = risk_filter[risk_filter['ic_quintile'] == 5]

def risk_ts():
    qtiles = pd.concat([risk['ANN_VOL_FCAST'].groupby(level=1).quantile(i) for i in [.25, .5, .75]], axis=1)
    qtiles.index = qtiles.index.map(lambda x:sort_order[x])
    ax = qtiles.sort_index().plot( legend=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

def cash_exp():
    #cash exposure
    #note use submissions rolled forward
    exposure = risk.groupby(axis=0,level=1).mean()[['CASH_NET_EXP','CASH_GROSS_EXP']].sort_index()
    exposure.columns = ['Net Exposure', 'Gross Exposure']
    exposure.index = exposure.index.map(lambda x:sort_order[x])
    ax = exposure.sort_index().plot(kind='bar',ylabel='Exposure')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

def risk_decomp():
    risk_decomp = risk.groupby(axis=0,level=1).mean()[['VAR_%ALL',
           'VAR_%FACTOR', 'VAR_%SPEC', 'VAR_%COV']]
    risk_decomp.columns=['M6M Factor','Other Factors','Specific Risk','Covariance']
    risk_decomp.index = risk_decomp.index.map(lambda x:sort_order[x])
    ax = risk_decomp.sort_index().plot(kind='bar',stacked=True)
    ax.set_ylabel('Allocation')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    #active decisions only
    risk_decomp = risk[risk['ActiveDecision']==1].groupby(axis=0, level=1).mean()[['VAR_%ALL',
                                                        'VAR_%FACTOR', 'VAR_%SPEC', 'VAR_%COV']]
    risk_decomp.columns = ['M6M Factor', 'Other Factors', 'Specific Risk', 'Covariance']
    risk_decomp.index = risk_decomp.index.map(lambda x: sort_order[x])
    ax = risk_decomp.sort_index().plot(kind='bar', stacked=True, title='Portfolio Risk Decomposition')
    ax.set_ylabel('Allocation')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

def rev_opt_anal():
    #create the asset ranks for each submission date
    asset_rtns = pd.concat([daily_rtn.reindex(submission_os_periods[i]).sum() for i in submission_os_periods.keys()], axis = 1)
    asset_ranks = quant_rank(asset_rtns,5)
    #asset_ranks = asset_ranks.drop(0, axis=1)
    rsort_order = {sort_order[i]: i for i in sort_order.keys()}
    asset_ranks.columns = asset_ranks.columns.map(lambda x: rsort_order[x])

    rev_opt_rank = quant_rank(reverse_alpha.T, 5)
    rev_opt_rank = rev_opt_rank.T

    all_scores_ex_trial=all_scores.loc[[i for i in risk.index if i[1] != 'Trial Period']]
    rev_opt_rank_ex_trial=rev_opt_rank.loc[[i for i in risk.index if i[1] != 'Trial Period']]


    rev_opt_corr_act = pd.Series(
        [pd.concat([all_scores_ex_trial.loc[i], rev_opt_rank_ex_trial.loc[i]], axis=1).corr().iloc[0, 1]
         for i in all_scores_ex_trial.index],
        index=all_scores_ex_trial.index)

    rev_opt_corr_act.describe()

    comp_scores = pd.concat([all_scores.stack(), rev_opt_rank_ex_trial.stack()], axis=1)
    sma.OLS(comp_scores[0], sma.add_constant(comp_scores[1].fillna(3))).fit().summary()

    act_corr_RTN = pd.Series(
        [pd.concat([all_scores.loc[i], asset_ranks[i[1]]], axis=1).corr().iloc[0, 1] for i in all_scores.index],
        index=all_scores.index)

    rev_opt_corr_RTN = pd.Series(
        [pd.concat([rev_opt_rank.loc[i], asset_ranks[i[1]]], axis=1).corr().iloc[0, 1] for i in all_scores.index],
        index=all_scores.index)

    top_rev_opt_rank = quant_rank(reverse_alpha.T, 5)
    top_rev_opt_rank = rev_opt_rank.T


    rev_opt_corr_act = pd.Series(
        [pd.concat([all_scores.loc[i], rev_opt_rank.loc[i]], axis=1).corr().iloc[0, 1] for i in all_scores.index],
        index=all_scores.index)

def IC_quintiles():
    # IC groupby
    risk.loc[[i for i in risk.index if i[1]!='Trial Period']].groupby('ic_quintile').median()[
        ['ic', 'ANN_VOL_FCAST', 'OptVol_M', 'ann_return', 'ann_return_opt_M', 'ann_ir', 'ann_ir_opt_M']]

def RPS_top_groupby():
    #portfolio performance for highest RPS winners
    risk.loc(axis=0)[rps_top.index, :][['ANN_VOL_FCAST', 'OptVol_M', 'ann_vol', 'ann_vol_opt_M', 'ann_return',
                                        'ann_return_opt_M', 'ann_ir', 'ann_ir_opt_M']].median()
    risk.loc(axis=0)[rps_top.index,:][['ANN_VOL_FCAST','OptVol_M','ann_vol','ann_vol_opt_M','ann_return',
                                       'ann_return_opt_M','ann_ir','ann_ir_opt_M']].median()

def IR_top_groupby():
    #portfolio performance for highest IR winners
    risk.loc(axis=0)[ir_top.index,:][['ANN_VOL_FCAST','OptVol_M','ann_vol','ann_vol_opt_M','ann_return',
                                   'ann_return_opt_M','ann_ir','ann_ir_opt_M']].median()

def rev_opt_top():

    top_risk_rev_opt, opt_wts_low_ic_top, opt_wts_med_ic_top, opt_wts_high_ic_top, reverse_alpha_top = \
        run_opt_fn(all_decisions_subset, risk, reverse_alpha, tgt=True)

def misc_code():
    pass
    #sd='Trial Period'
    #loadings_d = loadings[sd]
    #spec_vol_d = spec_risk[sd]
    #fcov_d = fcov[sd]
    #d=pd.Series(0.0,index=loadings_d.columns)
    #d['IVV']=1
    #portfolio_loadings = d[loadings_d.columns]@loadings_d.T
    #vol = np.sqrt(portfolio_loadings@fcov_d@portfolio_loadings + d[loadings_d.columns]@np.diag((spec_vol_d**2).squeeze())@d[loadings_d.columns])
    #key = ics['IC'].sort_values(ascending=False).index[0]
    #pd_key = key[1]
    #raw_scores = scores.loc[key]
    #actual_wts = decisions.loc[key]
    #opt(('0b1bfb9c', '11th Submission'))
