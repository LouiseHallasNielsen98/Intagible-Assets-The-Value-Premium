import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('Final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['sector_2'] == 3010].copy()
pd.set_option('display.max_columns', None)


### Intangibles

##Depreciation rate  
δ_RD = 0.15  
δ_SGA = 0.20  

##Growth rate
def average_growth(df, variable, id_col='sec_id', time_col='date'):
    df_sorted = df.sort_values(by=[id_col, time_col])
    df_sorted['growth'] = df_sorted.groupby(id_col)[variable].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    df_clean = df_sorted[np.isfinite(df_sorted['growth'])]
    return df_clean['growth'].mean()


g_RD = average_growth(df,variable='rd_value')
g_SGA = average_growth(df,variable='sga_value')

df[['rd_value', 'sga_value', 'goodwill_value']] = df[['rd_value', 'sga_value', 'goodwill_value']].fillna(0)

def compute_initial_intangible(group):
    first_sga = group['sga_value'].iloc[0]
    first_rd = group['rd_value'].iloc[0]
    
    # Calculate the components of intangibles
    O_i0 = (0.50 * first_sga / (g_SGA + δ_SGA))
    K_i0 = (0.70 * first_rd / (g_RD + δ_RD))
    int_i0 = O_i0 + K_i0
    
    # Return the components as separate columns
    return pd.Series({'O_i0': O_i0, 'K_i0': K_i0, 'INT_i0': int_i0})

int_initial = df.groupby('sec_id', group_keys=False)[['sga_value', 'rd_value']].apply(compute_initial_intangible)
df = df.merge(int_initial, on='sec_id', how='left')
df['INT'] = 0.0  
df['O_cap'] = 0.0 
df['K_cap'] = 0.0

for sec_id in df['sec_id'].unique():
    sec_data = df[df['sec_id'] == sec_id].copy()

    # Initialize the first value of INT
    sec_data.loc[sec_data.index[0], 'INT'] = float(sec_data['INT_i0'].iloc[0])
    sec_data.loc[sec_data.index[0], 'O_cap'] = float(sec_data['O_i0'].iloc[0])
    sec_data.loc[sec_data.index[0], 'K_cap'] = float(sec_data['K_i0'].iloc[0])

    for t in range(1, len(sec_data)):
        prev_row = sec_data.iloc[t-1]
        curr_row = sec_data.iloc[t]

        # Ensure that a new int is only calculated if the value changes to account for the ffill
        inputs = ['rd_value', 'sga_value']
        has_changed = any(
            not np.isclose(curr_row[var], prev_row[var], atol=1e-8) 
            for var in inputs
        )

        if has_changed:
            sga_t = curr_row['sga_value']
            rd_t = curr_row['rd_value']
            prev_O_cap = prev_row['O_cap']
            prev_K_cap = prev_row['K_cap']

            O_cap_new = prev_O_cap * (1 - δ_SGA) + (0.50 * sga_t)
            K_cap_new = prev_K_cap * (1 - δ_RD) + (0.70 * rd_t)

            new_INT = O_cap_new + K_cap_new

            sec_data.loc[sec_data.index[t], 'INT'] = new_INT
            sec_data.loc[sec_data.index[t], 'O_cap'] = O_cap_new 
            sec_data.loc[sec_data.index[t], 'K_cap'] = K_cap_new
        else:
            
            sec_data.loc[sec_data.index[t], 'INT'] = prev_row['INT']
            sec_data.loc[sec_data.index[t], 'O_cap'] = prev_row['O_cap']
            sec_data.loc[sec_data.index[t], 'K_cap'] = prev_row['K_cap']


    df.loc[sec_data.index, 'INT'] = sec_data['INT'].values
    df.loc[sec_data.index, 'O_cap'] = sec_data['O_cap'].values
    df.loc[sec_data.index, 'K_cap'] = sec_data['K_cap'].values

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['INT'] = df['INT'].fillna(0)
df['INT_to_MV'] = df['INT'] / df['market_cap']
firm_yearly_avg = df.groupby(['sec_id', 'year'])['INT_to_MV'].mean().reset_index()
avg_intangibles_df = firm_yearly_avg.groupby('year')['INT_to_MV'].mean().reset_index(name='avg_INT_to_MV_per_firm')
avg_intangibles_df.to_excel('avg_intangibles_to_mv_per_year_3010.xlsx', index=False)

