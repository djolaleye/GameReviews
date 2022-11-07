#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 07:59:20 2022

@author: deji
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import pickle



class game_predict():
    def __init__(self, model, scaler):
        with open('game_model', 'rb') as game_model, open('game_scaler', 'rb') as game_scaler:
            self.model = pickle.load(game_model)
            self.scaler = pickle.load(game_scaler)
            self.data = None
    
    def load_clean(self, data_steam, data_RAWG):
        game_data = pd.read_csv(data_steam, delimiter=',')
        
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
        
        
        
        rawg_data = pd.read_csv(data_RAWG, delimiter=',')

        rawg_data = rawg_data.drop(columns=['slug', 'website', 'playtime', 'achievements_count',
                                'suggestions_count', 'game_series_count', 'platforms', 'developers',
                                'publishers', 'esrb_rating', 'added_status_yet', 'added_status_owned', 
                                'added_status_beaten', 'added_status_toplay', 'added_status_dropped', 
                                'added_status_playing', 'genres', 'updated',
                                'rating_top', 'id'])

        rawg_data = rawg_data[rawg_data.tba != True]
        rawg_data = rawg_data.drop(columns='tba')

        rawg_data['released'] = pd.to_datetime(rawg_data['released'])
        rawg_data = rawg_data[rawg_data.released > '1999-12-31']
        rawg_data = rawg_data.drop(columns='released')

        rawg_data = rawg_data[rawg_data.ratings_count != 0]

        rawg_data = rawg_data.rename(columns={'name' : 'Title'})

        # rawg_data.isnull().sum()        # replace null metacritic scores with corresponding RAWG rating
        rawg_data_new = rawg_data.dropna(subset=['metacritic'])
        rawg_data_new = rawg_data_new.loc[~(rawg_data_new['rating'] == 0.0),:]
        
        
        
        result = pd.merge(rawg_data, game_data_clean, how='inner', on='Title')
        
        result['Reviews'] = result['Reviews_total'] + result['reviews_count']
        result.drop(columns=['Reviews_total', 'reviews_count'], inplace = True)
        
        q = result['Reviews'].quantile(0.99)
        result = result[result['Reviews']<q]
        
        
        '''
        Feature selection
        
        '''
        
        result.drop(columns=['Reviews_day7', 'Reviews_day30', 'Reviews_day90', 'Year'], inplace=True)
        result = result.rename(columns={'metacritic' : 'Metacritic', 'rating' : 'Rawg_rating', 
                                        'ratings_count' : 'Rawg_ratings_count'})
        
        # log transformations
        result['log_revenue'] = np.log(result['Revenue_est'])
        result['log_reviews'] = np.log(result['Reviews'])
        
        self.preprocessed_data = result.copy()
        self.data = self.scaler.transform(result)
        
    def predicted_outcomes(self):
        if (self.data is not None):
            self.preprocessed_data['Prediction'] = self.model.predict(self.data)
            return self.preprocessed_data
            
            
            
            
            
            
            
            
            
            
            
            
        