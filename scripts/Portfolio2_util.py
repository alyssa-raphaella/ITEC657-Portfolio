import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

import calendar
import datetime
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn import linear_model, utils
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score

# energydata DataFrame
energydata = pd.read_csv('data/energydata_complete.csv', index_col='date', parse_dates=True)
energydata.index = energydata.index.tz_localize('utc')

# Add DateTime Columns in DataFrame
energydata['Month'] = energydata.index.month
energydata['Month'] = energydata['Month'].apply(lambda x: calendar.month_abbr[x])
energydata['Date'] = energydata.index.date
energydata['Date'] = pd.to_datetime(energydata['Date'])
energydata['Day'] = energydata['Date'].dt.day_name()
energydata['Hour'] = energydata.index.hour
energydata['Week'] = energydata['Date'].dt.week
energydata = energydata.drop('Date',
                axis = 1)
#NSM - since lights and appliances are measure Wh, instead of NSM (s), changed it to hours
energydata['NSM'] = (energydata.index.second / 3600) + (energydata.index.minute / 60) + energydata.index.hour
#Weekend = True, Weekday = False
energydata['Week Status'] = ((pd.DatetimeIndex(energydata.index).dayofweek) // 5 == 1).astype(int)

### HEATMAP FUNCTION ###
# variables for heatmap function
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
energydata_hm_one = [] # new DataFrame holder
energydata_hm_two = [] # new DataFrame holder
energydata_hm_three = [] # new DataFrame holder
energydata_hm_four = [] # new DataFrame holder

def heatmap_func(one, two, three, four):
    fig, (ax1, ax2) = plt.subplots(1,2) # multiple heatmaps side by side
    fig, (ax3, ax4) = plt.subplots(1,2) # multiple heatmaps side by side
    week_num = energydata[energydata['Week']==one]
    for x in range(7):
        day_week = week_num[week_num['Day']==weekdays[x]]
        for y in range(24):
            hour_day = day_week[day_week['Hour']==y]
            hour_consump = hour_day['Appliances'].sum()
            energydata_hm_one.append((one, x, y, hour_consump))
    heatmap_one = pd.DataFrame(energydata_hm_one)
    heatmap_one.columns=['Week', 'Day of Week', 'Hour of Day', 'Consumption']
    heatmap_one.set_index(['Hour of Day', 'Day of Week']).unstack('Day of Week')
    pv  = pd.pivot_table(heatmap_one, values='Consumption', index=['Hour of Day'], columns=['Day of Week'])
    sns.heatmap(pv, 
                square=True, 
                xticklabels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                ax = ax1,
                cmap="Blues")
    week_num = energydata[energydata['Week']==two]
    for x in range(7):
        day_week = week_num[week_num['Day']==weekdays[x]]
        for y in range(24):
            hour_day = day_week[day_week['Hour']==y]
            hour_consump = hour_day['Appliances'].sum()
            energydata_hm_two.append((one, x, y, hour_consump))
    heatmap_two = pd.DataFrame(energydata_hm_two)
    heatmap_two.columns=['Week', 'Day of Week', 'Hour of Day', 'Consumption']
    heatmap_two.set_index(['Hour of Day', 'Day of Week']).unstack('Day of Week')
    pv  = pd.pivot_table(heatmap_two, values='Consumption', index=['Hour of Day'], columns=['Day of Week'])
    sns.heatmap(pv, 
                square=True, 
                xticklabels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                ax = ax2,
                cmap="Blues")
    week_num = energydata[energydata['Week']==three]
    for x in range(7):
        day_week = week_num[week_num['Day']==weekdays[x]]
        for y in range(24):
            hour_day = day_week[day_week['Hour']==y]
            hour_consump = hour_day['Appliances'].sum()
            energydata_hm_three.append((one, x, y, hour_consump))
    heatmap_three = pd.DataFrame(energydata_hm_three)
    heatmap_three.columns=['Week', 'Day of Week', 'Hour of Day', 'Consumption']
    heatmap_three.set_index(['Hour of Day', 'Day of Week']).unstack('Day of Week')
    pv  = pd.pivot_table(heatmap_three, values='Consumption', index=['Hour of Day'], columns=['Day of Week'])
    sns.heatmap(pv, 
                square=True, 
                xticklabels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                ax = ax3,
                cmap="Blues")
    week_num = energydata[energydata['Week']==four]
    for x in range(7):
        day_week = week_num[week_num['Day']==weekdays[x]]
        for y in range(24):
            hour_day = day_week[day_week['Hour']==y]
            hour_consump = hour_day['Appliances'].sum()
            energydata_hm_four.append((one, x, y, hour_consump))
    heatmap_four = pd.DataFrame(energydata_hm_four)
    heatmap_four.columns=['Week', 'Day of Week', 'Hour of Day', 'Consumption']
    heatmap_four.set_index(['Hour of Day', 'Day of Week']).unstack('Day of Week')
    pv  = pd.pivot_table(heatmap_four, values='Consumption', index=['Hour of Day'], columns=['Day of Week'])
    sns.heatmap(pv, 
                square=True, 
                xticklabels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                ax = ax4,
                cmap="Blues")

### MEAN ABSOLUTE PERCENTAGE ERROR ### 
def mean_absolute_percentage_error(y_train, predicted): 
    y_train, predicted = np.array(y_train), np.array(predicted)
    return np.mean(np.abs((y_train - predicted) / y_train)) * 100

def mean_absolute_percentage_error(y_test, predicted): 
    y_test, predicted = np.array(y_test), np.array(predicted)
    return np.mean(np.abs((y_test - predicted) / y_test)) * 100

### TRAINING AND TESTING ###
## normalization and splitting ##
all_cols = np.array(['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 
                 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T_out', 'Press_mm_hg', 
                 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'NSM'])

train = pd.read_csv('data/training.csv', index_col='date', parse_dates=True)
train.index = train.index.tz_localize('utc')
train = train.drop(['WeekStatus', 'Day_of_week'], axis=1)
train_norm = (train - train.mean())/train.std()
X_train = train_norm[all_cols]
y_train = train_norm['Appliances']

test = pd.read_csv('data/testing.csv', index_col='date', parse_dates=True)
test.index = test.index.tz_localize('utc')
test = test.drop(['WeekStatus', 'Day_of_week'], axis=1)
test_norm = (test - test.mean())/test.std()
X_test = test_norm[all_cols]
y_test = test_norm['Appliances']

### RFE ###
model = linear_model.LinearRegression()
selector = RFE(model, 9) #chose 9 features
selector = selector.fit(X_train,y_train)

supp = selector.get_support()
RFE_cols = all_cols[supp]
RFE_coeff = selector.estimator_.coef_
RFE_data = {'Feature':RFE_cols, 'Coefficient': RFE_coeff}
RFE_df = pd.DataFrame(RFE_data)

### STANDARDIZE DATA ###
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### MODELS ###
model_list = [linear_model.LinearRegression(), 
              RandomForestRegressor(n_estimators=20, random_state=0), 
              GradientBoostingRegressor(n_estimators=20, learning_rate=0.25, max_depth=13, random_state=0),
              SVR(kernel = "rbf")]
model_name = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR']

stats = []

for x in range(4):
    model = model_list[x]
    model.fit(X_train, y_train)
    predicted = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, predicted)
    R2_train = r2_score(y_train, predicted)
    MAE_train = mean_absolute_error(y_train, predicted)
    MAPE_train = mean_absolute_percentage_error(y_train, predicted)
    predicted = model.predict(X_test)
    MSE_test = mean_squared_error(y_test, predicted)
    R2_test = r2_score(y_test, predicted)
    MAE_test = mean_absolute_error(y_test, predicted)
    MAPE_test = mean_absolute_percentage_error(y_test, predicted)
    model = model_name[x]
    stats.append((model, MSE_train, MSE_test, R2_train, R2_test, MAE_train, MAE_test, MAPE_train, MAPE_test))
    stats_df = pd.DataFrame.from_records(stats)
    stats_df.columns = ['Model', 'MSE_Train', 'MSE_Test', 'R2_Train', 'R2_Test', 'MAE_Train', 'MAE_Test', 'MAPE_Train', 'MAPE_Test']
    stats_df.index = stats_df['Model']
    stats_df = stats_df.drop('Model',
                axis = 1)
    
