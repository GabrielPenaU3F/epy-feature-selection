# INPUT: dataset_2.0.csv
# OUTPUT: dataset_2.1.csv

# Organize the column names, remove unused columns and convert data types
# Assign the real peak number
# Normalize and remove old columns and population

import numpy as np
import pandas as pd


# Organization
dataset = pd.read_csv('../../../../resources/data/cases/dataset_v2/dataset_2.0.csv')
dataset = dataset.drop(columns=['index', 'peak_date', 'iso_code', 'continent', 'total_cases', 'new_cases_smoothed',
                      'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million',
                      'new_cases_smoothed_per_million', 'total_deaths_per_million', 'new_deaths_per_million',
                      'new_deaths_smoothed_per_million', 'icu_patients', 'icu_patients_per_million',
                      'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
                      'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
                      'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed',
                      'new_tests_smoothed_per_thousand', 'tests_units', 'total_vaccinations', 'people_vaccinated',
                      'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'new_vaccinations_smoothed',
                      'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
                      'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
                      'new_vaccinations_smoothed_per_million', 'new_people_vaccinated_smoothed',
                      'new_people_vaccinated_smoothed_per_hundred', 'hospital_beds_per_thousand',
                      'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
                      'excess_mortality', 'excess_mortality_cumulative_per_million'])
dataset = dataset.convert_dtypes(convert_integer=False)

# Real peak conversion
new_n_pico = np.array([])
for loc in dataset['location'].unique():
    loc_rows = dataset[dataset['location'] == loc]
    peaks = np.arange(1, len(loc_rows) + 1)
    new_n_pico = np.concatenate((new_n_pico, peaks))

dataset.drop(columns=['n_pico'], inplace=True)
dataset.insert(loc=1, column='n_peak', value=new_n_pico.astype('int64'))

# Normalizations
population = dataset['population'].to_numpy()
normalized_peak = (dataset['peak_magnitude'].to_numpy() / population) * 10000
normalized_hosp_patients = (dataset['hosp_patients'].to_numpy() / population) * 10000
normalized_total_tests = (dataset['total_tests'].to_numpy() / population) * 10
normalized_new_tests = (dataset['new_tests'].to_numpy() / population) * 1000
normalized_deaths = (dataset['total_deaths'].to_numpy() / population) * 10000

dataset = dataset.drop(columns=['peak_magnitude', 'hosp_patients', 'total_tests', 'new_tests', 'total_deaths',
                                'population'])
dataset.insert(loc=2, column='peak_magnitude_norm', value=normalized_peak)
dataset.insert(loc=3, column='hosp_patients_norm', value=normalized_hosp_patients)
dataset.insert(loc=4, column='total_tests_norm', value=normalized_total_tests)
dataset.insert(loc=5, column='new_tests_norm', value=normalized_new_tests)
dataset.insert(loc=6, column='total_deaths_norm', value=normalized_deaths)

# Out
dataset.to_csv('../../../resources/data/dataset_v2/dataset_2.1.csv', sep=',', encoding='utf-8', index=False)
