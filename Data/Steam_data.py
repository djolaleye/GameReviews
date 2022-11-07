#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:15:22 2022

@author: deji

Review scores from steam via steamspy & Game Data Crunch 
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set()

pd.options.display.max_columns = None

game_data = pd.read_csv('steam_trends.csv')

game_data = game_data.drop(columns =['Steam Page', 'Modified Tags', 'Tags', 'name_slug', 'App ID'])

game_data_clean = game_data.rename(columns={'Reviews Total' : 'Reviews_total', 'Reviews Score Fancy' : 'Review_score',
                                               'Release Date' : 'Release_date', 'Reviews D7' : 'Reviews_day7', 'Reviews D30' : 'Reviews_day30',
                                               'Reviews D90' : 'Reviews_day90', 'Launch Price' : 'Price', 'Revenue Estimated' : 'Revenue_est'})

# Change price data to float values

game_data_clean['Price'] = game_data_clean['Price'].str.replace('$', '')
game_data_clean['Price'] = game_data_clean['Price'].astype('float')


# Change revenue estimate data to float values

game_data_clean['Revenue_est'] = game_data_clean['Revenue_est'].str.replace('$', '')
game_data_clean['Revenue_est'] = game_data_clean['Revenue_est'].str.replace(',', '')
game_data_clean['Revenue_est'] = game_data_clean['Revenue_est'].astype('float')

# Change review score data to float values

game_data_clean['Review_score'] = game_data_clean['Review_score'].str.replace('%', '')
game_data_clean['Review_score'] = game_data_clean['Review_score'].astype('float')

# Change release date data to date values, change null values in Release_date to 9999-01-01, and separate out year of release

game_data_clean['Release_date'] = pd.to_datetime(game_data_clean['Release_date'])
game_data_clean = game_data_clean[game_data_clean['Release_date'] > '1999-12-31']
game_data_clean['Release_date'].fillna('9999-01-01', inplace = True)
game_data_clean['Year'] = pd.DatetimeIndex(game_data_clean['Release_date']).year
game_data_clean = game_data_clean.drop(columns='Release_date')

# dealing with null values
# any nulls in the dataframe ?
game_data_clean.isnull().sum()

# Delete rows with null Title, or Review score

game_data_clean = game_data_clean.dropna(subset=['Title'])
game_data_clean = game_data_clean.dropna(subset='Review_score')

# Remove tm, registered tm, copyright symbols
game_data_clean['Title'] = game_data_clean['Title'].str.replace(u"\u2122", '')
game_data_clean['Title'] = game_data_clean['Title'].str.replace(u"\u00AE", '')
game_data_clean['Title'] = game_data_clean['Title'].str.replace(u"\u00A9", '')

# Drop games with no reviews

game_data_clean['Reviews_total'].fillna(0, inplace = True)
game_data_clean = game_data_clean[game_data_clean.Reviews_total != 0]

# Drop rows with null values in review score, Reviews_day7/30/90 to zero,
# Price to the average price, and drop bottom 1% of Revenue


game_data_clean['Reviews_day7'].fillna(0, inplace = True)
game_data_clean['Reviews_day30'].fillna(0, inplace = True)
game_data_clean['Reviews_day90'].fillna(0, inplace = True)

price_var = sum(game_data_clean['Price'])
length = len(game_data_clean['Price'])

game_data_clean['Price'].fillna((price_var/length), inplace = True)

q = game_data_clean['Revenue_est'].quantile(0.01)
game_data_clean = game_data_clean[game_data_clean['Revenue_est']>q]

# Try and limit games to USA / English releases
# Remove non english alphabet characters

rows_to_drop = []

for i in game_data_clean['Title']:
    punctuation = ['.', '-', '!', ':', ';', ' ', '_']
    if i.isascii() :
        continue
    elif any(punct in i for punct in punctuation):
        continue
    else:
        ri = game_data_clean[game_data_clean['Title'] == i].index.item()
        rows_to_drop.append(ri)

game_data_clean = game_data_clean.drop(rows_to_drop, axis=0)



rawg_data = pd.read_csv('RAWG.csv')

result = pd.merge(rawg_data, game_data_clean, how='inner', on='Title')

result['Reviews'] = result['Reviews_total'] + result['reviews_count']
result.drop(columns=['Reviews_total', 'reviews_count'], inplace = True)

q = result['Reviews'].quantile(0.99)
result = result[result['Reviews']<q]


'''
Feature selection

'''

result_cor = result.corr()

result.drop(columns=['Reviews_day7', 'Reviews_day30', 'Reviews_day90', 'Year'], inplace=True)
result = result.rename(columns={'metacritic' : 'Metacritic', 'rating' : 'Rawg_rating', 
                                'ratings_count' : 'Rawg_ratings_count'})

# log transformations

result['log_revenue'] = np.log(result['Revenue_est'])
result['log_reviews'] = np.log(result['Reviews'])

plt.scatter(result['log_reviews'], result['log_revenue'])
plt.title('Log Reviews v. Log Revenue')
plt.show()

result['log_price'] = np.log(result['Price'])

plt.scatter(result['log_price'], result['log_revenue'])
plt.title('Log Price v. Log Revenue')
plt.show()


plt.scatter(result['Metacritic'], result['log_revenue'])
plt.title('Metacritic vs Log Revenue')
plt.show()


plt.scatter(result['Rawg_rating'], result['log_revenue'])
plt.title('RAWG rating vs Log Revenue')
plt.show()


# Create model using 'reviews_total, price,  metacritic, RAWG rating' features
x = result[['log_reviews', 'log_price', 'Metacritic', 'Rawg_rating']]
y = result['log_revenue']


scaler = StandardScaler()
scaler.fit(x)
scaled_inputs = scaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, y, test_size=0.2, random_state=20)


reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)


# Evaluate model's ability

plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=10)
plt.ylabel('Predictions (y_hat)', size=10)
plt.show()

sns.distplot(y_train - y_hat)

reg_summary = pd.DataFrame(x.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_



'''
Model Testing

'''

y_hat_test = reg.predict(x_test)

plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)', size=10)
plt.ylabel('Predictions (y_hat)', size=10)
plt.show()

sns.distplot(y_test - y_hat_test)

reg_test_summary = pd.DataFrame(np.exp(y_hat_test), columns=['Predicted Revenue'])
y_test = y_test.reset_index(drop=True)
reg_test_summary['Target'] = np.exp(y_test)
reg_test_summary['Residual'] = reg_test_summary['Target'] - reg_test_summary['Predicted Revenue']
reg_test_summary['Difference in %'] = np.absolute(reg_test_summary['Residual'] / reg_test_summary['Target']*100)



import pickle

with open('game_model', 'wb') as file:
    pickle.dump(reg, file)

with open('game_scaler', 'wb') as file:
    pickle.dump(scaler, file)



'''

Model has solid explanatory power regarding video game revenue. 
- R^2 training = 0.999, testing = 0.999
- 75% predicted w/in 6% 
- Lower targets -> higher error
Predictive utility is low, as this model relies heavily on total reviews, 
which is not helpful business-wise. Next steps would be to find different, more preemptive metrics to include. 

Potentially: hours to beat (content), wishlists, trailer views, website/social media interactions

Other further work: focus on certain genres, release windows, and platforms
'''



