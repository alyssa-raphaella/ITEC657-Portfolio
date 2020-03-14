import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

strava = pd.read_csv('data/strava_export.csv', index_col='date', parse_dates=True)
strava.index = strava.index.tz_convert('Australia/Sydney')

cheetah = pd.read_csv('data/cheetah.csv', skipinitialspace=True)
cheetah.index = pd.to_datetime(cheetah['date'] + ' ' + cheetah['time'])
cheetah.index = cheetah.index.tz_localize('Australia/Sydney')

table_merge=pd.merge(cheetah, strava, left_index=True, right_index=True, how='inner')

table_merge['Month']=table_merge.index.month
table_merge['Year']=table_merge.index.year
Summary=[]
MY=['Jan2017', 'Jan2018', 'Jan2019', 'Feb2017', 'Feb2018', 'Feb2019', 'Mar2017', 'Mar2018', 'Mar2019', 'Apr2017',
   'Apr2018', 'Apr2019', 'May2017', 'May2018', 'May2019', 'Jun2017', 'Jun2018', 'Jun2019', 'Jul2017', 'Jul2018', 'Jul2019',
   'Aug2017', 'Aug2018', 'Aug2019', 'Sep2017', 'Sep2018', 'Sep2019', 'Oct2017', 'Oct2018', 'Oct2019', 'Nov2017', 'Nov2018',
   'Nov2019', 'Dec2017', 'Dec2018', 'Dec2019']
mon=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
yr=[2017, 2018, 2019]

for x in mon:
    for y in yr:
        dt_table=table_merge[(table_merge.Month==x) & (table_merge.Year==y)]
        dt_d=dt_table['distance'].sum()
        dt_T=dt_table['TSS'].sum()
        dt_AV=dt_table['Average Speed'].mean()
        Summary.append((dt_d, dt_T, dt_AV))
        #MY.append((x,y))
        y=y+1
    x=x+1

table_merge['Day']=table_merge.index.day
category=['Ride', 'Race', 'Workout']

def func(x,y):
    dt_table=table_merge[(table_merge.Month==x) & (table_merge.Year==y)]
    dt_cat=dt_table[dt_table['workout_type']=='Ride']
    dt_ride=dt_cat.groupby('Day')['distance'].sum()
    dt_cat=dt_table[dt_table['workout_type']=='Race']
    dt_race=dt_cat.groupby('Day')['distance'].sum()
    dt_cat=dt_table[dt_table['workout_type']=='Workout']
    dt_wk=dt_cat.groupby('Day')['distance'].sum()
    df_summary=pd.concat([dt_ride, dt_race, dt_wk], axis=1)
    df_summary.columns=['Ride', 'Race', 'Workout']
    return df_summary.plot.bar(figsize=(8,4), width=1.0, title='Distance of rides per Day')