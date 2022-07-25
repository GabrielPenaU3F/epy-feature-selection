import numpy as np
import pandas as pd

dataset = pd.read_csv('../../resources/data/dataset_v1/dataset_1.2.csv')

new_n_pico = np.array([])
for loc in dataset['location'].unique():
    loc_rows = dataset[dataset['location'] == loc]
    peaks = np.arange(1, len(loc_rows) + 1)
    new_n_pico = np.concatenate((new_n_pico, peaks))

dataset.drop(columns=['n_pico'], inplace=True)
dataset.insert(loc=3, column='n_peak', value=new_n_pico.astype(int))

dataset.to_csv('../../resources/data/dataset_binned_v3.csv', sep=',', encoding='utf-8', index=False)
