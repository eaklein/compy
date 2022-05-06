#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:23:40 2022

@author: diesel
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/c/Users/Avram/Downloads/spotify.csv')
df['weight'] = 1/np.sqrt(df['Ranking'])

dance_wgt = {}
for city in df['City'].unique():
    df_city = df.loc[df['City'] == city]
    score = df_city['Danceability']*df_city['weight']
    score_norm = score / sum(df_city['weight'])
    print(f'{city}: {score_norm:.3f} {score}')
