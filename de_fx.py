import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import os
import yaml
import time
pd.set_option('display.max_columns', 500)
with open("configs.yml") as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

quote_style = configs['quote_style']

american_style = quote_style['american']
euro_style = quote_style['eu']
one_bps = quote_style['one_bps']
ten_bps = quote_style['ten_bps']
hundred_bps = quote_style['hundred_bps']


spot = pd.DataFrame()
fwd1M_pts = pd.DataFrame()
fwd1M = pd.DataFrame()
for root, dirs, files in os.walk(configs["data"], topdown=False):
    for name in files:
        if "spot" in name:
            df = pd.read_csv(os.path.join(root, name), index_col="Date", parse_dates=True).sort_index()
            df.columns = df.columns.str.lstrip('Line(Q').str.rstrip('=)')
            if len(spot) ==0:
                spot = df
            else:
                spot = spot.merge(df, how='outer', left_index=True, right_index=True)
        if "fwdp" in name:
            df = pd.read_csv(os.path.join(root, name), index_col="Date", parse_dates=True).sort_index()
            df.columns = df.columns.str.replace(" ","").str.lstrip('Line(Q').str.rstrip('=)')
            ## odd columns in TWD, MXN, BRL,KRW
            if name in ['TWD_fwdp.csv', 'MXN_fwdp.csv']:
                df = df[['fwdp']].rename({'fwdp': f"{name[:3]}1M"}, axis=1)
            if name == "BRL_fwdp.csv":
                df =df[['Unnamed:1']].rename({'Unnamed:1': f"{name[:3]}1M"}, axis=1)
            if len(fwd1M_pts) ==0:
                fwd1M_pts = df
            else:
                fwd1M_pts = fwd1M_pts.merge(df, how='outer', left_index=True, right_index=True)

        # print(name)
        # print(os.path.join(root, name))

fwd1M_pts = fwd1M_pts.ffill()

for spot_col in spot.columns:
    if spot_col in one_bps:
        fwd1M[spot_col] = fwd1M_pts[f"{spot_col}1M"]*0.0001 + spot[spot_col]
    elif spot_col in ten_bps:
        fwd1M[spot_col] = fwd1M_pts[f"{spot_col}1M"]*0.001 + spot[spot_col]
    elif spot_col in hundred_bps:
        fwd1M[spot_col] = fwd1M_pts[f"{spot_col}1M"] * 0.01 + spot[spot_col]
    else:
        print(spot_col)
    if spot_col in euro_style:
        spot[spot_col] = 1/spot[spot_col]
        fwd1M[spot_col] = 1/fwd1M[spot_col]

# print(spot.describe())
# print(fwd1M.describe())
fwd1M.to_csv(f"{configs['de']['fwd1M']}.csv")
spot.to_csv(f"{configs['de']['spot']}.csv")

fwd1M.to_pickle(f"{configs['de']['fwd1M']}.pkl")
spot.to_pickle(f"{configs['de']['spot']}.pkl")