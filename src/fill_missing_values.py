import pandas as pd
from sklearn.impute import KNNImputer

# Cases
# dataset = pd.read_csv('../resources/data/cases/dataset_v2/dataset_2.2.csv')

# Deaths
dataset = pd.read_csv('../resources/data/deaths/dataset_deaths_2.2.csv')

# Fill the vaccination columns with zeros
dataset['vaccination_1_dose'] = dataset['vaccination_1_dose'].fillna(0)
dataset['vaccination_full'] = dataset['vaccination_full'].fillna(0)

# Impute the rest of the values with KNN

location_col = dataset['location'].values
numerical_df = dataset.drop(columns=['location'])

imputer = KNNImputer(n_neighbors=3, weights="uniform")
imputed_dataset = pd.DataFrame(imputer.fit_transform(numerical_df), columns=numerical_df.columns)
imputed_dataset.insert(loc=0, column='location', value=location_col)

# Cases
# imputed_dataset.to_csv('../resources/data/dataset_v2/dataset_2.3.csv', sep=',', encoding='utf-8', index=False)

# Deaths
imputed_dataset.to_csv('../resources/data/deaths/dataset_deaths_2.3.csv', sep=',', encoding='utf-8', index=False)
