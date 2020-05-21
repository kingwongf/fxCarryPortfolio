import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import os
import yaml
import time
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)





class carry_backtest:
    def __init__(self, configs_loc):
        with open(configs_loc) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)
        self.fwd1M = pd.read_pickle(f"{self.configs['de']['fwd1M']}.pkl").sort_index().ffill()
        self.spot = pd.read_pickle(f"{self.configs['de']['spot']}.pkl").sort_index().ffill()

        ## daily returns
        # self.log_ret = np.log(self.spot).diff(1)
        self.spot_ret = self.spot.pct_change(1).fillna(0)


        ## approx. 90days as 3 months
        self.real_vol = self.spot_ret.rolling(90).std()
        self.carry = np.log(self.spot / self.fwd1M)



        ## carry cov_matrix and expected carry, rolling 90 days, expanding 90 days, or bootstrap if wanted
        self.carry_cov = self.carry.rolling(90).cov()
        self.carry_mean = self.carry.rolling(90).mean()

        self.rebalance = pd.read_csv(self.configs["rebalance"])
        self.rebalance.index = pd.to_datetime(self.rebalance['Date'])  ## maybe should check if market opens
        self.spot['rebalance'] = pd.to_datetime(self.rebalance['Date'])

        self.hist_index = None

        start_date = self.rebalance['Date'].iloc[0]

        # print(self.carry_mean.loc[start_date:].describe())

    def w_mthd1(self, carry_row,m):
        w = pd.Series(index=carry_row.index, name=carry_row.name)
        carry_row = carry_row.rank().sort_values(ascending=False).dropna()

        long = carry_row[:m].index
        short = carry_row[-m:].index
        w.loc[long] = 1/m
        w.loc[short] = -1/m
        return w
    def w_mthd2(self,carry_row, m, real_vol_row):
        w = pd.Series(index=carry_row.index, name=carry_row.name)
        carry_row = (carry_row/real_vol_row).rank().sort_values(ascending=False).dropna()
        long = carry_row[:m].index
        short = carry_row[-m:].index
        w.loc[long] = 1 / m
        w.loc[short] = -1 / m
        return w
    def w_mthd3(self, carry_cov):
        carry_cov = carry_cov.dropna(axis=1, how='all').dropna(axis=0, how='all')
        col_names = carry_cov.columns
        inv_covar = np.linalg.inv(carry_cov)
        u = np.ones(len(carry_cov))
        return pd.Series(np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u)), index=col_names)

    def w_mthd4(self, carry_cov, mean_log_ret, tgt_vol=0.10, long_short=True):
        ## TODO add bootstrap if you have time
        carry_cov = carry_cov.dropna(axis=1, how='all').dropna(axis=0, how='all')
        mean_log_ret = mean_log_ret.dropna(axis=0)
        w_bounds = (-1,1) if long_short else (0,1)
        eff = EfficientFrontier(expected_returns=mean_log_ret, cov_matrix=carry_cov, weight_bounds= w_bounds)
        w = eff.efficient_risk(target_risk=tgt_vol, market_neutral=long_short)
        return pd.Series(w)



    def mainBacktester(self):
        init_index = 1000
        init_USD_cash = 1000
        indices_cols = [f"index_mthd{i}" for i in range(1,5)]
        port_cols = [f"port_mthd{i}" for i in range(1,5)]

        ##  first rebalancing date is 2000-01-10 00:00:00, so index starts at 1999-12-10 00:00:00

        indices_df = pd.DataFrame([[init_index for _ in range(len(indices_cols))]], index=[pd.to_datetime("1999-12-10")], columns=indices_cols)

        port_df = pd.DataFrame([[init_USD_cash for _ in range(len(port_cols))]], index=[pd.to_datetime("2000-01-10")], columns=port_cols)
        for date, spot_row in self.spot.iterrows():
            curr_indices = indices_df.iloc[-1]
            curr_portfolios = port_df.iloc[-1]
            if not pd.isnull(spot_row['rebalance']):
                ## rebalancing indices
                w_1 = self.w_mthd1(self.carry.loc[date], 3)
                w_2 = self.w_mthd2(self.carry.loc[date], 3, self.real_vol.loc[date])
                w_3 = self.w_mthd3(self.carry_cov.loc[date])
                w_4 = self.w_mthd4(self.carry_cov.loc[date], self.carry_mean.loc[date], tgt_vol=0.10, long_short=False)

                li_index = [index*(np.sum(w * self.carry.loc[date]) + 1) for index, w in zip(curr_indices, [w_1, w_2, w_3, w_4])]
                indices_df = indices_df.append(pd.Series(li_index, name=date, index=indices_cols))
            ## tracking portfolio returns performance
            if date >= curr_portfolios.name:
                li_port = [trading_capital*(np.sum(w * self.spot_ret.loc[date]) +1) for trading_capital, w in zip(curr_portfolios, [w_1, w_2, w_3, w_4])]
                port_df = port_df.append(pd.Series(li_port, name=date, index=port_cols))

        port_df.to_pickle(f"{self.configs['de']['portfolios']}.pkl")
        indices_df.to_pickle(f"{self.configs['de']['indices']}.pkl")

    def mainPerformanceAnalysis(self):
        indices_df = pd.read_pickle(f"{self.configs['de']['indices']}.pkl")
        port_df = pd.read_pickle(f"{self.configs['de']['portfolios']}.pkl")
        # print(indices_df)
        rf = pd.read_csv(self.configs['rf'], index_col='Date', parse_dates=True, dayfirst=True)
        # rf.index = pd.to_datetime(rf.index)
        # print(rf)

        fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6,1, sharex=True)
        # ax0.set_supp('Strategy Performance')

        ## Indices Plot
        indices_df.plot(ax=ax0)
        ax0.set_ylabel("Index Value")
        ax0.legend()

        ## Portfolios Equity Curve
        port_df.plot(ax=ax1)
        ax1.set_ylabel("Portfolio Value (USD)")
        ax1.legend()



        ## Rolling Yearly Return
        (port_df.pct_change(126).add(1)).pow(2).sub(1).plot(ax=ax2, lw=0.5, legend=False)
        ax2.set_ylabel("126 days Annualised Return")
        #ax2.legend()

        ## Realised Risk of Portfolios
        port_df.pct_change().rolling(126).std().multiply(16).plot(ax=ax3, legend=False)
        ax3.set_ylabel("126 days Annualised Vol")
        #ax3.legend()


        ## Rolling Max Drawdowns
        def dd(ts):
            return np.min(ts / np.maximum.accumulate(ts)) - 1

        port_df.rolling(126).apply(dd).plot(ax=ax4, legend=False)
        ax4.set_ylabel("126 days Max Drawdown")
        #ax4.legend()

        ## Rolling Sharpe
        rolling_Sharpe = np.sqrt(252)*(port_df.pct_change(1) - rf.loc[port_df.index].values*0.01*1/252).rolling(126).mean() / port_df.pct_change(1).rolling(126).std()
        rolling_Sharpe.plot(ax=ax5, lw=0.5, legend=False)
        ax5.set_ylabel("126 days Sharpe")
        #ax5.legend()

        plt.subplots_adjust(0.06, 0.04, 0.98, 0.99, 0.03, 0.00)
        plt.show()
        exit()




        ## Annualised Yearly Sharpe
        ## Need to downsample to yearly basis for annualised Sharpe
        yr_ret_port_df = port_df.resample('1y').first().pct_change(1)
        annual_rf = rf.resample('1y').mean().loc[yr_ret_port_df.index] * 0.01
        annual_std = port_df.pct_change(252).resample('1y').std()

        annual_Sharpe = (yr_ret_port_df - annual_rf.values) / annual_std
        print(annual_Sharpe)






         #print(annualised_Sharpe)



'''
## method 1
m = 10
rebalance_carry = carry.loc[rebalance.index]

print(rebalance_carry)
exit()
def index_w_mthd1(carry,m):
    

    :param carry:
    :param m: no. of fx pairs to long/short
    :return:
    
    rank_df = carry.rank(axis=1, method='first')
    no_ranks = len(rank_df.columns)
    print(rank_df)
    assert no_ranks >= m * 2
    for i in range(no_ranks, no_ranks - m, -1):
        print(f"+ve {i} w")
        rank_df[rank_df == i] = 1 / m
    for i in range(1, m + 1, 1):
        print(f"-ve {i} w")
        rank_df[rank_df == i] = -1 / m
    rank_df.where(rank_df < 1, 0, inplace=True)
    return rank_df


print(index_w_mthd1(rebalance_carry, 3)) ## .sum(axis=1)
# rebalance_carry.rank(axis=0)
# index_weight =
'''