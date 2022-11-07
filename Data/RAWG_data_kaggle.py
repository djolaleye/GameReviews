#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:17:17 2022

@author: deji
"""
import Steam_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pd.options.display.max_columns = None
pd.options.display.max_rows = None

rawg_data = pd.read_csv('game_info.csv')

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


# rawg_data_new.to_csv('RAWG.csv', index = False)



