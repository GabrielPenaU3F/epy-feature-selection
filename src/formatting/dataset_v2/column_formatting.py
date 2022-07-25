# INPUT: dataset_2.0.csv
# OUTPUT: dataset_2.1.csv

# Organize the column names, remove unused columns and convert data types

import pandas as pd


dataset = pd.read_csv('../../../resources/data/dataset_v2/dataset_2.0.csv')
dataset = dataset.drop(columns=['index', 'peak_date', 'iso_code', 'continent', 'total_cases', 'new_cases_smoothed',
                      'total_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million',
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

dataset.to_csv('../../../resources/data/dataset_v2/dataset_2.1.csv', sep=',', encoding='utf-8', index=False)
